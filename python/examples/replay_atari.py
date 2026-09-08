"""CPU replay of a complete frozen vector run, with task outcomes and optional video.

Replays every recorded action in fresh environments. It does not construct or
train Kindle. A movie shows every episode of the chosen stream, not selected
successes. RAM is used only by the post-hoc Qbert completion observer.
"""

import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path
import shutil
import subprocess

import ale_py
import ale_py._ale_py as ale_native
import gymnasium as gym

import atari
from atari_tasks import EpisodeTask, ROM_SHA256
from audit_atari import read_run, require, sha256
from check_atari_adapter import rom_identity
from record_atari import AtariVideo


class TaskMonitor(gym.Wrapper):
    def __init__(self, environment, name, rom_sha256):
        super().__init__(environment)
        self.name, self.rom_sha256 = name, rom_sha256
        self.task = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.task = EpisodeTask(self.name, self.rom_sha256)
        return result

    def step(self, action):
        result = self.env.step(action)
        ram = self.unwrapped.ale.getRAM().tolist() if self.name == 'ALE/Qbert-v5' else None
        self.task.advance(float(result[1]), ram)
        return result


def verify_replay_identity(header, manifest, rom):
    require(header['environment'] == manifest['environment']
            and header['environment'] in ROM_SHA256, 'wrong declared replay game')
    require(rom['sha256'] == ROM_SHA256[header['environment']], 'unsupported replay ROM')
    require(header['ale_py_version'] == ale_py.__version__ == '0.12.1', 'changed ALE version')
    require(header['wrapper_sha256'] == sha256(atari.__file__), 'changed source wrapper')
    require(header['atari_protocol'] == 'published' and header['action_repeat'] == 4
            and header['sticky_actions'] == 0 and header['noop_max'] == 0
            and header['full_action_space'] and header['max_episode_frames'] == 100000,
            'unsupported replay preprocessing')
    for path in (Path(rom['path']), Path(ale_native.__file__), Path(atari.__file__)):
        # The source wrapper may live in another checkout, with identical bytes.
        expected = header['wrapper_sha256'] if path == Path(atari.__file__) else manifest['pins'].get(str(path))
        require(expected is not None and sha256(path) == expected, f'changed replay input: {path}')


def replay_rows(source, environments, monitors):
    episodes = []
    for line in source:
        row = json.loads(line)
        if row['event'] == 'transition':
            results = [environment.step(action) for environment, action in zip(environments, row['actions'])]
            expected = dict(rewards=[result[1] for result in results],
                            terminated=[result[2] for result in results], truncated=[result[3] for result in results],
                            executed_action_frames=[environment.executed_action_frames for environment in environments])
            for key, value in expected.items():
                require(row[key] == value, f'replay diverged at action {row["run_step"]}: {key}')
        elif row['event'] == 'episode':
            stream = row['stream']
            result = monitors[stream].task.result(row['terminated'], row['truncated'])
            require(result['episode_score'] == row['episode_return'], 'task score differs from episode ledger')
            end = environments[stream].executed_action_frames
            episodes.append(dict(**row, task_outcome=result, first_frame=end - result['episode_frames'], last_frame=end))
        elif row['event'] == 'reset':
            for stream in row['streams']:
                environments[stream].reset()
        elif row['event'] == 'learner':
            raise ValueError('replay source contains learning')
    partial = [dict(stream=stream, **monitor.task.result(False, False)) for stream, monitor in enumerate(monitors)]
    return episodes, partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--video-stream', type=int, default=0)
    args = parser.parse_args()
    targets = [args.output] + ([args.video] if args.video else [])
    require(len(set(path.resolve() for path in targets)) == len(targets), 'output paths must differ')
    for path in targets:
        require(not path.exists() and path.parent.is_dir(), 'outputs must be fresh paths in existing directories')
    run = read_run(args.log)
    header = run['start']
    require(header['mode'] in ('evaluate_sample', 'evaluate_greedy') and run['accounting']['updates'] == 0,
            'reconstruct only a complete frozen evaluation')
    require(0 <= args.video_stream < header['num_envs'], 'invalid video stream')
    ffmpeg = shutil.which('ffmpeg') if args.video else None
    require(not args.video or ffmpeg is not None, 'ffmpeg is required for video')
    manifest = json.loads(args.source_manifest.read_text())
    gym.register_envs(ale_py)
    rom = rom_identity(header['environment'])
    verify_replay_identity(header, manifest, rom)
    environments, monitors = [], []
    recorder = None
    with ExitStack() as stack:
        for stream, seed in enumerate(header['environment_seeds']):
            raw = stack.enter_context(closing(gym.make(header['environment'], frameskip=1,
                                                       repeat_action_probability=0.0, full_action_space=True)))
            if args.video and stream == args.video_stream:
                recorder = AtariVideo(raw, args.video, ffmpeg)
                stack.callback(recorder.close)
                raw = recorder
            monitor = TaskMonitor(raw, header['environment'], rom['sha256'])
            environment = atari.DreamerAtariPreprocessing(monitor, noop_max=0, max_episode_frames=100000)
            require(list(environment.action_meanings) == header['action_meanings'], 'changed action vocabulary')
            environment.reset(seed=seed)
            environments.append(environment)
            monitors.append(monitor)
        with args.log.open() as source:
            require(json.loads(next(source)) == header, 'source header changed during replay')
            episodes, partial = replay_rows(source, environments, monitors)
        frames = [environment.executed_action_frames for environment in environments]
        require(frames == run['end']['executed_action_frames'], 'replayed total frames differ')
        require([environment.emulator_resets for environment in environments] == run['end']['emulator_resets'],
                'replayed emulator resets differ')
        require(sha256(args.log) == run['sha256'], 'source log changed during replay')
        if recorder:
            require(recorder.frame_count == frames[args.video_stream], 'recording frame count differs')
    video = None
    if recorder:
        video = dict(path=str(args.video.resolve()), sha256=sha256(args.video), stream=args.video_stream,
                     frames=recorder.frame_count, fps=60, raw_frames_sha256=recorder.frame_sha256.hexdigest(),
                     ffmpeg_version=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0])
    result = dict(protocol='kindle-atari-task-replay-v1', source_log=str(args.log.resolve()),
                  source_log_sha256=run['sha256'], source_header=header, source_accounting=run['accounting'],
                  source_manifest=str(args.source_manifest.resolve()), source_manifest_sha256=sha256(args.source_manifest),
                  rom=rom, ale_native_sha256=sha256(ale_native.__file__),
                  wrapper_sha256=sha256(atari.__file__), replay_script_sha256=sha256(__file__),
                  observer_sha256=sha256(Path(__file__).with_name('atari_tasks.py')),
                  agent_constructed=False, learner_updates=0, full_trajectory_replayed=True,
                  verified='all recorded actions, rewards, boundaries, resets and actual frame counts',
                  original_pixels_available_for_comparison=False, video=video, episodes=episodes, partial=partial,
                  task_successes=sum(episode['task_outcome']['episode_success'] for episode in episodes),
                  reliability_assessed=False)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), completed_episodes=len(episodes),
                          task_successes=result['task_successes'], agent_constructed=False)))


if __name__ == '__main__':
    main()

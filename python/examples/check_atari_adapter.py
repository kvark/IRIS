"""CPU-only Atari adapter check: interleaved streams versus fresh serial replay.

Actions are uniformly random coverage, not a learned policy. No Kindle agent
is constructed. This checks images, rewards, boundaries and emulator clocks;
it does not test learning, GPU batching or game competence.
"""

import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import ale_py
import ale_py._ale_py as ale_native
from ale_py import roms
import gymnasium as gym
import numpy as np
import PIL

import atari


PANEL = ("Pong", "Breakout", "Boxing", "Freeway", "Seaquest", "Frostbite", "Qbert", "PrivateEye")
STREAMS = 2


def make_environment(name):
    protocol = atari.ATARI_PROTOCOLS["published"]
    return atari.DreamerAtariPreprocessing(
        gym.make(name, frameskip=1, repeat_action_probability=0.0,
                 full_action_space=protocol.full_action_space),
        noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames)


def rom_identity(name):
    path = Path(roms.get_rom_path(gym.spec(name).kwargs["game"]))
    return dict(path=str(path.resolve()), sha256=atari.sha256_file(path))


def frame_hash(frame):
    if not isinstance(frame, np.ndarray) or frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
        raise ValueError("expected a 64x64 RGB8 observation")
    return hashlib.sha256(frame.tobytes()).hexdigest()


def reset_record(env, stream, step, seed=None):
    before = env.executed_action_frames
    frame, info = env.reset(seed=seed)
    if env.executed_action_frames != before or info["noop_count"] != 0 or env.reset_noop_frames != 0:
        raise ValueError("reset changed the declared action/no-op clocks")
    return dict(event="reset", stream=stream, step=step, seed=seed,
                frame_sha256=frame_hash(frame), executed_action_frames=env.executed_action_frames,
                emulator_resets=env.emulator_resets)


def step_record(env, stream, step, action):
    before_wrapper = env.executed_action_frames
    before_ale = env.unwrapped.ale.getFrameNumber()
    frame, reward, terminated, truncated, info = env.step(action)
    frames = env.executed_action_frames - before_wrapper
    if frames != info["frame_number"] - before_ale or not 1 <= frames <= atari.ATARI_ACTION_REPEAT:
        raise ValueError("wrapper and ALE frame clocks disagree")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError("reward must be finite and numeric")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("boundary flags must be boolean")
    return dict(event="transition", stream=stream, step=step, action=action,
                reward=float(reward), terminated=terminated, truncated=truncated,
                frame_sha256=frame_hash(frame), action_frames=frames,
                executed_action_frames=env.executed_action_frames)


def verify_serial(name, seeds, history):
    for stream, seed in enumerate(seeds):
        with closing(make_environment(name)) as env:
            generator = random.Random(seed ^ 0xA7A2_1000)
            for expected in history:
                if expected["stream"] != stream:
                    continue
                step = expected["step"]
                if expected["event"] == "reset":
                    actual = reset_record(env, stream, step, seed if step == 0 else None)
                else:
                    action = generator.randrange(env.action_space.n)
                    if type(expected["action"]) is not int or expected["action"] != action:
                        raise ValueError(f"serial replay mismatch: stream {stream}, step {step}, random action sequence")
                    actual = step_record(env, stream, step, expected["action"])
                if actual != expected:
                    fields = sorted(key for key in actual.keys() | expected.keys()
                                    if actual.get(key) != expected.get(key))
                    raise ValueError(f"serial replay mismatch: stream {stream}, step {step}, fields {fields}")


def check_game(name, steps_per_stream, seed, path):
    if steps_per_stream <= 0 or not 0 <= seed < 2**32:
        raise ValueError("steps must be positive and seed must fit an unsigned 32-bit integer")
    seeds = [(seed + stream * 1_000_003) % 2**32 for stream in range(STREAMS)]
    action_seeds = [value ^ 0xA7A2_1000 for value in seeds]
    history = []
    episodes = []
    partial_returns = [0.0] * STREAMS
    partial_lengths = [0] * STREAMS
    rewards = [0.0] * STREAMS
    positive_events = [0] * STREAMS
    negative_events = [0] * STREAMS
    started = time.monotonic()
    with path.open("x") as output, ExitStack() as stack:
        def emit(event):
            print(json.dumps(event, separators=(",", ":"), allow_nan=False), file=output, flush=True)

        envs = [stack.enter_context(closing(make_environment(name))) for _ in range(STREAMS)]
        meanings = list(envs[0].action_meanings)
        if any(list(env.action_meanings) != meanings or env.action_space.n != len(meanings) for env in envs):
            raise ValueError("streams have different action vocabularies")
        action_counts = [[0] * len(meanings) for _ in envs]
        generators = [random.Random(value) for value in action_seeds]
        emit(dict(event="run_start", protocol="kindle-atari-adapter-v1", environment=name,
                  mode="forced_random_adapter_diagnostic", agent_constructed=False,
                  streams=STREAMS, steps_per_stream=steps_per_stream, environment_seeds=seeds,
                  action_seeds=action_seeds, action_meanings=meanings, atari_protocol="published",
                  action_repeat=atari.ATARI_ACTION_REPEAT, full_action_space=True,
                  sticky_actions=0.0, noop_max=0,
                  max_episode_frames=atari.ATARI_PROTOCOLS["published"].max_episode_frames,
                  rom=rom_identity(name), ale_py_version=ale_py.__version__,
                  ale_native_sha256=atari.sha256_file(ale_native.__file__),
                  gymnasium_version=gym.__version__, numpy_version=np.__version__,
                  pillow_version=PIL.__version__, python_version=sys.version,
                  wrapper_sha256=atari.sha256_file(atari.__file__),
                  checker_sha256=atari.sha256_file(__file__), unix_time=time.time()))
        for stream, env in enumerate(envs):
            record = reset_record(env, stream, 0, seeds[stream])
            history.append(record)
            emit(record)
        for step in range(1, steps_per_stream + 1):
            for stream, env in enumerate(envs):
                action = generators[stream].randrange(len(meanings))
                record = step_record(env, stream, step, action)
                history.append(record)
                emit(record)
                action_counts[stream][action] += 1
                reward = record["reward"]
                rewards[stream] += reward
                positive_events[stream] += reward > 0
                negative_events[stream] += reward < 0
                partial_returns[stream] += reward
                partial_lengths[stream] += 1
                if record["terminated"] or record["truncated"]:
                    episode = dict(event="episode", stream=stream, step=step,
                                   episode_return=partial_returns[stream], episode_length=partial_lengths[stream],
                                   terminated=record["terminated"], truncated=record["truncated"])
                    episodes.append(episode)
                    emit(episode)
                    partial_returns[stream] = 0.0
                    partial_lengths[stream] = 0
                    reset = reset_record(env, stream, step)
                    history.append(reset)
                    emit(reset)
        collection_seconds = time.monotonic() - started
        frames = [env.executed_action_frames for env in envs]
        resets = [env.emulator_resets for env in envs]
        for stream in range(STREAMS):
            completed = [item for item in episodes if item["stream"] == stream]
            if (sum(action_counts[stream]) != steps_per_stream
                    or sum(item["episode_length"] for item in completed) + partial_lengths[stream] != steps_per_stream
                    or sum(item["episode_return"] for item in completed) + partial_returns[stream] != rewards[stream]
                    or resets[stream] != 1 + len(completed)):
                raise ValueError("episode, action, reward or reset ledger disagrees")
        stack.close()
        replay_started = time.monotonic()
        verify_serial(name, seeds, history)
        result = dict(event="run_end", environment=name, status="adapter_replay_verified",
                      agent_constructed=False, learner_updates=0, competence_evaluated=False,
                      collected_actions=STREAMS * steps_per_stream,
                      serial_replay_actions=STREAMS * steps_per_stream, executed_action_frames=frames,
                      emulator_resets=resets, action_counts=action_counts,
                      total_rewards=rewards, positive_reward_events=positive_events,
                      negative_reward_events=negative_events,
                      completed_episodes=len(episodes),
                      natural_episodes=sum(item["terminated"] and not item["truncated"] for item in episodes),
                      truncated_episodes=sum(item["truncated"] for item in episodes),
                      mean_completed_return=(sum(item["episode_return"] for item in episodes) / len(episodes)
                                             if episodes else None),
                      partial_returns=partial_returns, partial_lengths=partial_lengths,
                      collection_seconds=collection_seconds, serial_replay_seconds=time.monotonic() - replay_started,
                      verified=["RGB8 observation hashes", "executed actions", "raw rewards", "boundary flags",
                                "per-action ALE/wrapper clocks", "fresh serial replay of each interleaved stream"])
        emit(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--environments", nargs="+", default=[f"ALE/{game}-v5" for game in PANEL])
    parser.add_argument("--steps-per-stream", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8001)
    args = parser.parse_args()
    if args.steps_per_stream <= 0 or not 0 <= args.seed < 2**32:
        parser.error("steps must be positive and seed must fit an unsigned 32-bit integer")
    if len(set(args.environments)) != len(args.environments):
        parser.error("environments must be distinct")
    args.directory.mkdir(parents=True, exist_ok=False)
    gym.register_envs(ale_py)
    results = []
    for name in args.environments:
        path = args.directory / (name.replace("/", "-") + ".jsonl")
        try:
            result = check_game(name, args.steps_per_stream, args.seed, path)
        except Exception as error:
            result = dict(environment=name, status="failed", error=f"{type(error).__name__}: {error}")
        result.update(log=str(path), log_sha256=atari.sha256_file(path) if path.exists() else None)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    with (args.directory / "summary.json").open("x") as output:
        json.dump(results, output, indent=2, allow_nan=False)
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

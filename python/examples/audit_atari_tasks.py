"""Frozen Freeway, Breakout and Qbert task gates, not a campaign or seed-reliability certificate.

Requires the complete final train/evaluation ledger, checkpoint and CPU replay.
Match games remain in audit_atari.py. No acceptance rule is inferred from scores.
"""

import argparse
import json
import math
from pathlib import Path
import statistics

import ale_py._ale_py as ale_native

import atari_tasks
from atari_tasks import TASKS
from audit_atari import read_run, require, sha256, verify_checkpoint
from check_atari_adapter import rom_identity
import replay_atari


TASK_CRITERIA = {
    'ALE/Freeway-v5': dict(minimum_completed_episodes=20, minimum_success_fraction=0.90,
                          minimum_mean_return=25, maximum_truncated_episodes=0),
    'ALE/Breakout-v5': dict(minimum_completed_episodes=20, minimum_success_fraction=0.90,
                           minimum_mean_return=None, maximum_truncated_episodes=None),
    'ALE/Qbert-v5': dict(minimum_completed_episodes=20, minimum_success_fraction=0.90,
                        minimum_mean_return=15000, maximum_truncated_episodes=None),
}


def check_task(environment, task, completed):
    require(environment in TASK_CRITERIA, 'no supported task gate')
    require(task['task'] == TASKS[environment], 'wrong task observer')
    score, frames, milestone = task['episode_score'], task['episode_frames'], task['first_milestone_frame']
    require(type(score) in (int, float) and math.isfinite(score) and score >= 0 and score == int(score),
            'invalid task score')
    require(type(frames) is int and frames >= int(completed), 'invalid task frame count')
    require(milestone is None or type(milestone) is int and 1 <= milestone <= frames,
            'invalid milestone frame')
    terminal, truncated = task['terminated'], task['truncated']
    require(type(terminal) is type(truncated) is bool and (terminal or truncated) == completed,
            'invalid task boundary')
    require(task['eligible_completed_episode'] is completed, 'invalid task eligibility')
    if environment == 'ALE/Qbert-v5':
        cubes = task['max_initial_qbert_cubes']
        require(type(cubes) is int and 0 <= cubes <= 21 and (cubes == 21) == (milestone is not None),
                'inconsistent Qbert pyramid evidence')
    else:
        require(task['max_initial_qbert_cubes'] is None, 'Qbert evidence on another task')
        threshold = 25 if environment == 'ALE/Freeway-v5' else 864
        require((score >= threshold) == (milestone is not None), 'score and milestone disagree')
        require(environment != 'ALE/Breakout-v5' or score <= 864, 'invalid Breakout final score')
    success = completed and milestone is not None
    if environment == 'ALE/Freeway-v5':
        success = success and terminal and not truncated
    require(task['episode_success'] is success, 'invalid task-success flag')
    return success


def score_tasks(environment, episodes):
    require(environment in TASK_CRITERIA, 'no supported task gate')
    identities, returns, successes, natural, truncated = set(), [], 0, 0, 0
    for episode in episodes:
        stream, index = episode['stream'], episode['episode']
        require(type(stream) is type(index) is int and stream >= 0 and index >= 0,
                'invalid episode identity')
        require((stream, index) not in identities, 'duplicate completed episode')
        identities.add((stream, index))
        task = episode['task_outcome']
        successes += check_task(environment, task, True)
        require(episode['episode_return'] == task['episode_score']
                and episode['terminated'] is task['terminated']
                and episode['truncated'] is task['truncated'], 'task and episode ledger disagree')
        returns.append(task['episode_score'])
        natural += task['terminated'] and not task['truncated']
        truncated += task['truncated']
    criteria = TASK_CRITERIA[environment]
    count = len(episodes)
    mean = statistics.mean(returns) if count else None
    fraction = successes / count if count else None
    passed = (count >= criteria['minimum_completed_episodes']
              and fraction >= criteria['minimum_success_fraction']
              and (criteria['minimum_mean_return'] is None or mean >= criteria['minimum_mean_return'])
              and (criteria['maximum_truncated_episodes'] is None
                   or truncated <= criteria['maximum_truncated_episodes']))
    return dict(environment=environment, criteria=criteria, completed_episodes=count,
                natural_episodes=natural, truncated_episodes=truncated, task_successes=successes,
                mean_completed_return=mean, task_success_fraction=fraction, task_gate_passed=passed,
                score_scope='completed episodes of one frozen policy; partial tails excluded',
                reliability_assessed=False)


def check_replay(evaluation, replay):
    header = evaluation['start']
    require(replay['protocol'] == 'kindle-atari-task-replay-v1', 'unsupported replay protocol')
    require(Path(replay['source_log']).resolve() == Path(evaluation['path']).resolve()
            and replay['source_log_sha256'] == evaluation['sha256']
            and replay['source_header'] == header
            and replay['source_accounting'] == evaluation['accounting'], 'replay source differs')
    require(replay['agent_constructed'] is False and replay['learner_updates'] == 0
            and replay['full_trajectory_replayed'] is True, 'not a complete CPU replay')
    manifest = Path(replay['source_manifest'])
    require(sha256(manifest) == replay['source_manifest_sha256'], 'replay manifest changed')
    rom = rom_identity(header['environment'])
    require(rom == replay['rom'], 'replay ROM differs')
    replay_atari.verify_replay_identity(header, json.loads(manifest.read_text()), rom)
    require(replay['ale_native_sha256'] == sha256(ale_native.__file__)
            and replay['wrapper_sha256'] == header['wrapper_sha256']
            and replay['replay_script_sha256'] == sha256(replay_atari.__file__)
            and replay['observer_sha256'] == sha256(atari_tasks.__file__), 'replay implementation changed')
    require(len(replay['episodes']) == len(evaluation['episodes']), 'replay episode count differs')
    last_frames = [0] * header['num_envs']
    for actual, recorded in zip(replay['episodes'], evaluation['episodes']):
        require(all(actual[key] == value for key, value in recorded.items()), 'replay episode ledger differs')
        stream = recorded['stream']
        first, last = actual['first_frame'], actual['last_frame']
        require(type(first) is type(last) is int and first == last_frames[stream] and last > first
                and last - first == actual['task_outcome']['episode_frames'], 'replay episode frames differ')
        last_frames[stream] = last
    partial = replay['partial']
    require([task['stream'] for task in partial] == list(range(header['num_envs'])), 'missing partial streams')
    for stream, task in enumerate(partial):
        check_task(header['environment'], task, False)
        require(task['episode_score'] == evaluation['end']['partial_returns'][stream]
                and task['episode_frames'] == evaluation['end']['executed_action_frames'][stream] - last_frames[stream],
                'partial task ledger differs')
    require(replay['task_successes'] == sum(task['task_outcome']['episode_success'] for task in replay['episodes']),
            'replay success count differs')


def audit_final_tasks(training_path, evaluation_path, checkpoint, schema, replay_path):
    training, evaluation = read_run(training_path), read_run(evaluation_path)
    start, frozen = training['start'], evaluation['start']
    require(start['mode'] == 'train' and frozen['mode'] == 'evaluate_sample', 'wrong train/evaluation mode')
    require(start['starting_environment_step'] == start['starting_learner_step'] == 0
            and start['restored_checkpoint'] is None and training['accounting']['updates'] > 0,
            'training must be fresh and have updates')
    require(evaluation['accounting']['updates'] == 0, 'evaluation is not frozen')
    # Keep this identical to the pinned match auditor until its live consumer ends.
    for key in ('environment', 'protocol', 'atari_protocol', 'action_repeat', 'full_action_space',
                'noop_max', 'max_episode_frames', 'sticky_actions', 'action_meanings', 'ale_py_version',
                'config', 'model_provenance', 'native_extension_sha256', 'runner_sha256', 'wrapper_sha256',
                'trainable_parameter_counts', 'gpu_device', 'cpu_worker_threads'):
        require(start[key] == frozen[key], f'changed evaluation identity: {key}')
    replay = json.loads(Path(replay_path).read_text())
    check_replay(evaluation, replay)
    result = score_tasks(frozen['environment'], replay['episodes'])
    result['checkpoint'] = verify_checkpoint(checkpoint, training, evaluation, schema)
    result['training'] = {key: training[key] for key in ('path', 'sha256', 'accounting', 'end')}
    result['evaluation'] = {key: evaluation[key] for key in ('path', 'sha256', 'accounting', 'end')}
    result['replay'] = dict(path=str(replay_path), sha256=sha256(replay_path),
                          partial=replay['partial'], full_trajectory_replayed=True)
    result['campaign_declaration_verified'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'evaluation', 'checkpoint', 'schema', 'replay', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'output must be fresh')
    result = audit_final_tasks(args.train, args.evaluation, args.checkpoint, args.schema, args.replay)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), task_gate_passed=result['task_gate_passed'],
                          campaign_declaration_verified=False, reliability_assessed=False)))


if __name__ == '__main__':
    main()

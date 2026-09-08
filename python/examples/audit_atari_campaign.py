"""Verify the complete predeclared five-game, three-training-seed replication.

The declaration fixes all 15 run paths, complete config (without seed), header
identity, per-game budgets, criteria, source pins and the checkpoint schema.
This does not select a recipe, launch training, or count pilot models as repeats.
Untrained controls and the broader goal-completion audit remain separate.
"""

import argparse
import json
import math
from pathlib import Path

import ale_py._ale_py as ale_native
import kindle._vector_audit as vector_audit

import atari_tasks
import audit_atari as matches
import audit_atari_tasks as tasks
import check_atari_adapter
import replay_atari
from audit_atari import read_run, require, sha256


PROTOCOL = 'kindle-atari-five-replication-v1'
SEEDS = [1009, 2017, 3019]
CRITERIA = {**matches.MATCH_CRITERIA, **tasks.TASK_CRITERIA}
ATARI_HEADER = dict(protocol='kindle-vector-v2', atari_protocol='published',
                    full_action_space=True, action_repeat=4, sticky_actions=0.0,
                    noop_max=0, max_episode_frames=100000, ale_py_version='0.12.1',
                    policy_seed_rule='config.seed + stream (wrapping u64)')
HEADER_KEYS = set(ATARI_HEADER) | {
    'action_meanings', 'model_provenance', 'native_extension_sha256',
    'runner_sha256', 'wrapper_sha256', 'trainable_parameter_counts',
    'gpu_device', 'cpu_worker_threads',
}


def positive_integer(value):
    return type(value) is int and value > 0


def verify_declaration(declaration):
    require(declaration['protocol'] == PROTOCOL, 'unsupported replication declaration')
    require(declaration['training_seeds'] == SEEDS, 'changed fresh replication seeds')
    streams = declaration['training_streams']
    require(streams == declaration['evaluation_streams'] == 8
            and type(streams) is int, 'replication requires eight streams')
    seed = declaration['evaluation_seed']
    require(type(seed) is int and 0 <= seed < 2**32, 'invalid frozen environment seed')
    timestamp = declaration['declared_unix_time']
    require(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp > 0,
            'invalid declaration time')
    require(set(declaration['games']) == set(CRITERIA), 'all five declared games are required')
    for environment, game in declaration['games'].items():
        require(game['criteria'] == CRITERIA[environment], 'changed task acceptance criteria')
        for key in ('training_actions', 'evaluation_actions'):
            require(positive_integer(game[key]) and game[key] % streams == 0, 'invalid game action budget')
    header = declaration['header']
    require(set(header) == HEADER_KEYS, 'incomplete declared header identity')
    require(all(header[key] == value for key, value in ATARI_HEADER.items()), 'changed Atari protocol')
    require(header['model_provenance']['perception']['kind'] == 'levjepa', 'changed frontend')
    config = declaration['config']
    require('seed' not in config and config['action_count'] == 18
            and config['extrinsic_reward_scale'] == 1 and config['intrinsic_reward_scale'] == 0
            and config['visitation_bonus'] is False, 'changed seed/action/reward declaration')
    json.dumps(declaration, allow_nan=False)
    rows = declaration['runs']
    expected = {(game, seed) for game in CRITERIA for seed in SEEDS}
    require(len(rows) == len(expected)
            and {(row['environment'], row['seed']) for row in rows} == expected,
            'require exactly one run per game and fresh training seed')
    paths = [Path(row[key]).resolve() for row in rows
             for key in ('training', 'evaluation', 'checkpoint', 'replay')]
    require(len(set(paths)) == len(paths), 'reused run artifact path')
    require(not set(map(Path, declaration['pins'])).intersection(paths), 'output is a pinned input')


def verify_inputs(declaration):
    pins = declaration['pins']
    for path, expected in pins.items():
        require(Path(path).is_absolute() and sha256(path) == expected, f'changed declared input: {path}')
    for module in (matches, tasks, replay_atari, atari_tasks, check_atari_adapter, vector_audit):
        path = str(Path(module.__file__).resolve())
        require(pins.get(path) == sha256(path), f'unpinned auditor: {path}')
    for path in (Path(__file__).resolve(), Path(ale_native.__file__).resolve(),
                 *(Path(declaration['schema']) / name for name in
                   ('metadata.json', 'world.safetensors', 'behavior.safetensors', 'slow_value.safetensors'))):
        require(pins.get(str(path)) == sha256(path), f'unpinned audit dependency: {path}')
    header = declaration['header']
    for key, role in (('runner_sha256', 'runner'), ('wrapper_sha256', 'wrapper'),
                      ('native_extension_sha256', 'native')):
        require(pins[declaration['inputs'][role]] == header[key], f'declared executable differs: {role}')
    require(pins[declaration['inputs']['encoder']]
            == header['model_provenance']['perception']['checkpoint_sha256'], 'declared encoder differs')
    for game in CRITERIA:
        rom = check_atari_adapter.rom_identity(game)
        require(pins.get(rom['path']) == rom['sha256'] == atari_tasks.ROM_SHA256[game], 'unpinned game ROM')


def verify_run_declaration(declaration, row, training, evaluation):
    game = declaration['games'][row['environment']]
    config = dict(declaration['config'], seed=row['seed'])
    for run, mode, seed, budget in (
        (training, 'train', row['seed'], game['training_actions']),
        (evaluation, 'evaluate_sample', declaration['evaluation_seed'], game['evaluation_actions']),
    ):
        header = run['start']
        require(header['environment'] == row['environment'] and header['mode'] == mode,
                'wrong declared game or action mode')
        require(header['seed'] == seed and header['num_envs'] == 8, 'changed declared run seed/streams')
        require(header['steps'] == run['accounting']['actions'] == budget
                and run['accounting']['budget_complete'], 'changed or incomplete declared action budget')
        require(header['environment_seeds'] == [(seed + stream * 1_000_003) % 2**32 for stream in range(8)],
                'changed environment stream seed rule')
        require(header['config'] == config, 'changed complete declared training config')
        for key, value in declaration['header'].items():
            require(header[key] == value, f'changed declared header: {key}')
        timestamp = header['unix_time']
        require(type(timestamp) in (int, float) and math.isfinite(timestamp)
                and timestamp > declaration['declared_unix_time'], 'run predates its declaration')
    start = training['start']
    require(start['restored_checkpoint'] is None
            and start['starting_environment_step'] == start['starting_learner_step'] == 0
            and training['accounting']['updates'] > 0, 'not fresh training with updates')
    require(evaluation['accounting']['updates'] == 0, 'evaluation is not frozen')
    require(training['end']['unix_time'] < evaluation['start']['unix_time'], 'evaluation precedes final training')


def check_match_replay(evaluation, replay):
    """Bind a completed match replay; task check_replay is deliberately task-only."""
    header = evaluation['start']
    require(replay['protocol'] == 'kindle-atari-task-replay-v1'
            and replay['source_log_sha256'] == evaluation['sha256']
            and Path(replay['source_log']).resolve() == Path(evaluation['path']).resolve()
            and replay['source_header'] == header
            and replay['source_accounting'] == evaluation['accounting'], 'match replay source differs')
    require(replay['agent_constructed'] is False and replay['learner_updates'] == 0
            and replay['full_trajectory_replayed'] is True, 'not a complete CPU match replay')
    manifest_path = Path(replay['source_manifest'])
    require(sha256(manifest_path) == replay['source_manifest_sha256'], 'match replay manifest changed')
    rom = check_atari_adapter.rom_identity(header['environment'])
    require(replay['rom'] == rom, 'match replay ROM differs')
    replay_atari.verify_replay_identity(header, json.loads(manifest_path.read_text()), rom)
    for key, path in (('ale_native_sha256', ale_native.__file__), ('wrapper_sha256', replay_atari.atari.__file__),
                      ('replay_script_sha256', replay_atari.__file__), ('observer_sha256', atari_tasks.__file__)):
        require(replay[key] == sha256(path), 'match replay implementation changed')
    require(len(replay['episodes']) == len(evaluation['episodes']), 'match replay episode count differs')
    last_frames = [0] * header['num_envs']
    for actual, recorded in zip(replay['episodes'], evaluation['episodes']):
        require(all(actual[key] == value for key, value in recorded.items()), 'match replay episodes differ')
        stream = recorded['stream']
        task = actual['task_outcome']
        require(actual['first_frame'] == last_frames[stream] and actual['last_frame'] > actual['first_frame']
                and actual['last_frame'] - actual['first_frame'] == task['episode_frames'], 'match replay frames differ')
        require(task['task'] == atari_tasks.TASKS[header['environment']]
                and task['episode_score'] == recorded['episode_return']
                and task['terminated'] is recorded['terminated'] and task['truncated'] is recorded['truncated']
                and task['eligible_completed_episode'] is True
                and task['episode_success'] is (recorded['terminated'] and not recorded['truncated']
                                                and recorded['episode_return'] > 0),
                'match replay task differs')
        last_frames[stream] = actual['last_frame']
    require([tail['stream'] for tail in replay['partial']] == list(range(header['num_envs'])), 'missing match tails')
    for stream, tail in enumerate(replay['partial']):
        require(tail['terminated'] is False and tail['truncated'] is False
                and tail['eligible_completed_episode'] is False and tail['episode_success'] is False
                and tail['episode_score'] == evaluation['end']['partial_returns'][stream]
                and tail['episode_frames'] == evaluation['end']['executed_action_frames'][stream] - last_frames[stream],
                'match replay tail differs')
    require(replay['task_successes'] == sum(row['task_outcome']['episode_success'] for row in replay['episodes']),
            'match replay success count differs')


def audit_campaign(path):
    path = Path(path)
    declaration_hash = sha256(path)
    declaration = json.loads(path.read_text())
    verify_declaration(declaration)
    verify_inputs(declaration)
    results, identities = [], set()
    for row in declaration['runs']:
        training, evaluation = read_run(row['training']), read_run(row['evaluation'])
        verify_run_declaration(declaration, row, training, evaluation)
        replay = json.loads(Path(row['replay']).read_text())
        if row['environment'] in matches.MATCH_CRITERIA:
            check_match_replay(evaluation, replay)
            score = matches.score_matches(row['environment'], evaluation['episodes'])
            passed = score['mastery_passed']
        else:
            tasks.check_replay(evaluation, replay)
            score = tasks.score_tasks(row['environment'], replay['episodes'])
            passed = score['task_gate_passed']
        checkpoint = matches.verify_checkpoint(row['checkpoint'], training, evaluation, declaration['schema'])
        identity = tuple(checkpoint['tensor_sha256'][name] for name in ('world', 'behavior', 'slow_value'))
        require(identity not in identities, 'reused trained tensor state')
        identities.add(identity)
        results.append(dict(environment=row['environment'], seed=row['seed'], passed=passed, score=score,
                            checkpoint=checkpoint, training_sha256=training['sha256'],
                            evaluation_sha256=evaluation['sha256'], replay_sha256=sha256(row['replay'])))
    require(sha256(path) == declaration_hash, 'declaration changed during audit')
    verify_inputs(declaration)
    return dict(protocol=PROTOCOL, declaration=dict(path=str(path.resolve()), sha256=declaration_hash),
                results=results, campaign_declaration_verified=True, reliability_assessed=True,
                reliability_scope='all three fresh training seeds must pass each fixed final task gate',
                replication_passed=all(row['passed'] for row in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('declaration', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'output must be fresh')
    result = audit_campaign(args.declaration)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), replication_passed=result['replication_passed'])))


if __name__ == '__main__':
    main()

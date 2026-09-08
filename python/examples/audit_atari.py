"""Frozen Atari match scores and final-checkpoint checks, not a seed-reliability claim.

Only Pong and Boxing currently have implemented win rules. A positive score in
another Atari game is not enough to call it a win. Confidence intervals describe
one frozen policy's episodes, not independent training seeds.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
from safetensors import safe_open

from kindle._vector_audit import audit


MATCH_CRITERIA = {
    'ALE/Pong-v5': dict(minimum_natural_episodes=20, minimum_mean_return=15,
                        minimum_natural_win_fraction=0.90, maximum_truncated_episodes=0),
    'ALE/Boxing-v5': dict(minimum_natural_episodes=20, minimum_mean_return=50,
                          minimum_natural_win_fraction=0.90, maximum_truncated_episodes=0),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def wilson_interval(successes, count):
    if not count:
        return None
    z = statistics.NormalDist().inv_cdf(0.975)
    rate = successes / count
    scale = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def score_matches(environment, episodes):
    require(environment in MATCH_CRITERIA, 'no implemented game-specific win rule')
    bound = 21 if environment == 'ALE/Pong-v5' else 100
    for episode in episodes:
        value = episode['episode_return']
        require(type(value) in (int, float) and math.isfinite(value)
                and value == int(value) and abs(value) <= bound, 'invalid match return')
        require(type(episode['terminated']) is bool and type(episode['truncated']) is bool
                and (episode['terminated'] or episode['truncated']), 'invalid episode boundary')
        require(type(episode['stream']) is int and episode['stream'] >= 0, 'invalid stream')
    natural = [episode for episode in episodes if episode['terminated'] and not episode['truncated']]
    returns = [float(episode['episode_return']) for episode in natural]
    count = len(natural)
    wins = sum(value > 0 for value in returns)
    timeouts = sum(episode['truncated'] for episode in episodes)
    mean = statistics.mean(returns) if count else None
    rate = wins / count if count else None
    criteria = MATCH_CRITERIA[environment]
    # Resample whole stream histories, preserving within-stream dependence.
    streams = sorted(set(episode['stream'] for episode in natural))
    by_stream = [[episode['episode_return'] for episode in natural if episode['stream'] == stream]
                 for stream in streams]
    interval = None
    if len(streams) >= 2:
        generator = random.Random(20260908)
        samples = sorted(statistics.mean(value for history in generator.choices(by_stream, k=len(streams))
                                         for value in history) for _ in range(2000))
        interval = [samples[49], samples[1949]]
    passed = (count >= criteria['minimum_natural_episodes'] and mean >= criteria['minimum_mean_return']
              and rate >= criteria['minimum_natural_win_fraction']
              and timeouts <= criteria['maximum_truncated_episodes'])
    return dict(environment=environment, criteria=criteria, natural_episodes=count, natural_wins=wins,
                draws=sum(value == 0 for value in returns), losses=sum(value < 0 for value in returns),
                truncated_episodes=timeouts, mean_natural_return=mean, natural_win_fraction=rate,
                mean_return_stream_bootstrap_95=interval, bootstrap_streams=len(streams),
                natural_win_wilson_95=wilson_interval(wins, count),
                confidence_scope='conditional on this frozen policy, not training-seed reliability',
                mastery_passed=passed, reliability_assessed=False)


def read_run(path):
    accounting = audit(path)
    require(accounting['budget_complete'], 'incomplete declared run budget')
    episodes = []
    checkpoint = None
    with Path(path).open() as source:
        start = json.loads(next(source))
        for line in source:
            event = json.loads(line)
            if event['event'] == 'episode':
                episodes.append(event)
            elif event['event'] == 'checkpoint':
                checkpoint = event
            elif event['event'] == 'transition':
                require(all(pair[1] == 0 for pair in event['stored_rewards']),
                        'intrinsic reward in extrinsic-only experiment')
            end = event
    require(start['config']['extrinsic_reward_scale'] == 1
            and start['config']['intrinsic_reward_scale'] == 0
            and not start['config']['visitation_bonus'], 'changed reward recipe')
    require(end['reset_noop_frames'] == [0] * start['num_envs']
            and end['emulator_resets'] == [count + 1 for count in end['episode_counts']],
            'emulator reset/no-op ledger mismatch')
    return dict(path=str(path), sha256=sha256(path), start=start, end=end, checkpoint=checkpoint,
                accounting=accounting, episodes=episodes)


def verify_checkpoint(path, training, evaluation, schema):
    path, schema = Path(path), Path(schema)
    metadata = json.loads((path / 'metadata.json').read_text())
    template_metadata = json.loads((schema / 'metadata.json').read_text())
    start, final = training['start'], training['end']
    frozen = evaluation['start']
    require(metadata['format'] == template_metadata['format'] == 3, 'wrong checkpoint format')
    require(metadata['architecture'] == template_metadata['architecture'] == 'dreamerv3-visual-features',
            'wrong checkpoint architecture')
    require(metadata['config'] == start['config'] == frozen['config'], 'changed checkpoint config')
    require(metadata['collection_streams'] == start['num_envs'], 'changed training stream count')
    require(metadata['perception'] == start['model_provenance']['perception']
            == frozen['model_provenance']['perception'], 'changed perception')
    require(metadata['environment_step'] == final['environment_step'] == frozen['starting_environment_step']
            and metadata['learner_step'] == final['learner_step'] == frozen['starting_learner_step'],
            'not the declared final checkpoint')
    saved = training['checkpoint']
    require(saved is not None and saved['run_step'] == final['run_step']
            and saved['learner_step'] == final['learner_step'], 'missing final save')
    identity = dict(metadata_sha256=sha256(path / 'metadata.json'), tensor_sha256=metadata['tensor_sha256'])
    for recorded in (saved['identity'], frozen['restored_checkpoint']):
        require(recorded is not None and all(recorded[key] == value for key, value in identity.items()),
                'different saved/restored checkpoint files')
    for key in ('dreamerv3_revision', 'meganeura_revision', 'blade_revision', 'future_head_revision'):
        require(metadata[key] == template_metadata[key] == start['model_provenance'][key],
                f'changed model/backend identity: {key}')
    tensors = {}
    for name in ('world', 'behavior', 'slow_value'):
        current, template = path / f'{name}.safetensors', schema / f'{name}.safetensors'
        require(sha256(current) == metadata['tensor_sha256'][name]
                and sha256(template) == template_metadata['tensor_sha256'][name], 'damaged tensor file')
        with safe_open(current, framework='numpy') as model, safe_open(template, framework='numpy') as reference:
            require(model.keys() == reference.keys() and bool(model.keys()), 'incomplete tensor names')
            for key in model.keys():
                value, expected = model.get_tensor(key), reference.get_tensor(key)
                require(value.shape == expected.shape and value.dtype == expected.dtype, 'changed tensor schema')
                require(np.isfinite(value).all() and np.isfinite(expected).all(), 'non-finite tensor')
                if key.startswith('adam_v.'):
                    require(np.all(value >= 0), 'negative optimizer second moment')
            tensors[name] = len(model.keys())
    return dict(path=str(path), **identity, tensors=tensors, finite_and_complete=True)


def compare_final(training_path, evaluation_path, checkpoint, schema):
    training, evaluation = read_run(training_path), read_run(evaluation_path)
    start, frozen = training['start'], evaluation['start']
    require(start['mode'] == 'train' and frozen['mode'] == 'evaluate_sample', 'wrong train/evaluation mode')
    require(start['starting_environment_step'] == start['starting_learner_step'] == 0
            and start['restored_checkpoint'] is None and training['accounting']['updates'] > 0,
            'training must be fresh and have updates')
    require(evaluation['accounting']['updates'] == 0, 'evaluation is not frozen')
    for key in ('environment', 'protocol', 'atari_protocol', 'action_repeat', 'full_action_space',
                'noop_max', 'max_episode_frames', 'sticky_actions', 'action_meanings', 'ale_py_version',
                'config', 'model_provenance', 'native_extension_sha256', 'runner_sha256', 'wrapper_sha256',
                'trainable_parameter_counts', 'gpu_device', 'cpu_worker_threads'):
        require(start[key] == frozen[key], f'changed evaluation identity: {key}')
    result = score_matches(frozen['environment'], evaluation['episodes'])
    result['checkpoint'] = verify_checkpoint(checkpoint, training, evaluation, schema)
    result['training'] = {key: training[key] for key in ('path', 'sha256', 'accounting', 'end')}
    result['evaluation'] = {key: evaluation[key] for key in ('path', 'sha256', 'accounting', 'end')}
    return result


def verify_boxing_declaration(path, training, evaluation):
    manifest = json.loads(Path(path).read_text())
    require(manifest['protocol'] == 'kindle-boxing-ratio-pilot-v1', 'unknown campaign declaration')
    start, frozen = training['start'], evaluation['start']
    expected = dict(environment=manifest['environment'], protocol=manifest['vector_protocol'],
                    atari_protocol=manifest['atari_protocol'], full_action_space=manifest['full_action_space'],
                    action_repeat=manifest['action_repeat'], sticky_actions=manifest['sticky_actions'],
                    noop_max=manifest['noop_max'], max_episode_frames=manifest['max_episode_frames'])
    for key, value in expected.items():
        require(start[key] == frozen[key] == value, f'changed declared protocol: {key}')
    require(start['environment'] == 'ALE/Boxing-v5' and start['seed'] == start['config']['seed']
            and start['seed'] in manifest['training_seeds'], 'changed declared training seed/game')
    require(start['steps'] == manifest['training_actions']
            and start['num_envs'] == manifest['training_streams'], 'changed declared training budget')
    require(frozen['steps'] == manifest['frozen_evaluation_actions']
            and frozen['num_envs'] == manifest['frozen_evaluation_streams']
            and frozen['seed'] == manifest['frozen_environment_seed'], 'changed declared evaluation budget/seed')
    require(frozen['mode'] == 'evaluate_sample' and manifest['frozen_action_mode'] == 'sample'
            and evaluation['accounting']['updates'] == manifest['frozen_updates'] == 0,
            'changed declared evaluation mode')
    for header in (start, frozen):
        require(header['environment_seeds'] == [(header['seed'] + stream * 1_000_003) % 2**32
                                               for stream in range(header['num_envs'])], 'changed stream seed rule')
        require(header['policy_seed_rule'] == 'config.seed + stream (wrapping u64)', 'changed policy seed rule')
    for key in ('batch_size', 'batch_length', 'world_backprop_length', 'world_microbatch_size',
                'learning_rate', 'learning_rate_warmup', 'agc'):
        require(start['config'][key] == manifest[key], f'changed declared recipe: {key}')
    require(start['config']['train_ratio'] in manifest['ratio_order']
            and start['config']['model_size'] == 'size12_m' and manifest['model_size'] == '12m'
            and start['model_provenance']['perception']['kind'] == 'levjepa'
            and manifest['frontend'] == 'LeVJEPA', 'changed declared ratio/model/frontend')
    for key in ('reconstruction', 'future_prediction'):
        require(start['config']['loss_scales'][key] == manifest[key], 'changed declared world objective')
    for key, value in MATCH_CRITERIA['ALE/Boxing-v5'].items():
        require(manifest['boxing_mastery'][key] == value, 'changed declared mastery criterion')
    for name, expected_hash in manifest['pins'].items():
        require(sha256(name) == expected_hash, f'changed pinned campaign input: {name}')
    source = Path(manifest['source_worktree'])
    for key, name in (('runner_sha256', source / 'python/examples/atari_vector.py'),
                      ('wrapper_sha256', source / 'python/examples/atari.py')):
        require(start[key] == manifest['pins'][str(name)], f'changed declared executable: {key}')
    native_hashes = [value for name, value in manifest['pins'].items() if '/kindle/_native.' in name]
    require(native_hashes == [start['native_extension_sha256']], 'changed declared native extension')
    return dict(path=str(path), sha256=sha256(path), training_seed=start['seed'],
                train_ratio=start['config']['train_ratio'], declaration_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--schema', required=True, type=Path,
                        help='complete validated checkpoint on the same backend and model shape')
    parser.add_argument('--declaration', type=Path, help='also verify the pinned Boxing pilot protocol')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare_final(args.train, args.evaluation, args.checkpoint, args.schema)
    result['campaign_declaration'] = (verify_boxing_declaration(args.declaration, read_run(args.train),
                                                              read_run(args.evaluation))
                                      if args.declaration else None)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), mastery_passed=result['mastery_passed'],
                          reliability_assessed=False)))


if __name__ == '__main__':
    main()

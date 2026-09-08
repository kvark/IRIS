import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import audit_atari


def episode(score=60, stream=0, terminated=True, truncated=False):
    return dict(episode_return=score, stream=stream, terminated=terminated, truncated=truncated)


def test_boxing_win_is_not_a_positive_score_in_another_game():
    rows = [episode(60, stream=i % 8) for i in range(24)]
    result = audit_atari.score_matches('ALE/Boxing-v5', rows)
    assert result['mastery_passed']
    assert result['natural_wins'] == 24
    assert not result['reliability_assessed']
    assert result['mean_return_stream_bootstrap_95'] == [60, 60]
    with pytest.raises(ValueError, match='no implemented game-specific win rule'):
        audit_atari.score_matches('ALE/Freeway-v5', rows)


def test_draws_losses_timeouts_and_partial_episodes_do_not_become_wins():
    rows = [episode(60), episode(0), episode(-10), episode(100, truncated=True)]
    result = audit_atari.score_matches('ALE/Boxing-v5', rows)
    assert result['natural_episodes'] == 3
    assert result['natural_wins'] == result['draws'] == result['losses'] == 1
    assert result['truncated_episodes'] == 1
    assert result['mean_natural_return'] == pytest.approx(50 / 3)
    assert not result['mastery_passed']
    with pytest.raises(ValueError, match='invalid episode boundary'):
        audit_atari.score_matches('ALE/Boxing-v5', [episode(100, terminated=False)])


@pytest.mark.parametrize('rows', [
    [episode(100) for _ in range(19)],
    [episode(49) for _ in range(20)],
    [episode(100) for _ in range(17)] + [episode(-1) for _ in range(3)],
    [episode(100) for _ in range(20)] + [episode(100, truncated=True)],
])
def test_each_predeclared_boxing_gate_matters(rows):
    assert not audit_atari.score_matches('ALE/Boxing-v5', rows)['mastery_passed']


def test_empty_match_sample_is_not_a_result():
    result = audit_atari.score_matches('ALE/Pong-v5', [])
    assert result['mean_natural_return'] is result['natural_win_fraction'] is None
    assert result['mean_return_stream_bootstrap_95'] is result['natural_win_wilson_95'] is None
    assert not result['mastery_passed']


@pytest.mark.parametrize('score', [True, None, [], '50', float('nan'), float('inf'), 100.5, 101])
def test_invalid_match_returns_fail(score):
    with pytest.raises(ValueError, match='invalid match return'):
        audit_atari.score_matches('ALE/Boxing-v5', [episode(score)])


def test_pong_bound_and_single_stream_confidence():
    with pytest.raises(ValueError, match='invalid match return'):
        audit_atari.score_matches('ALE/Pong-v5', [episode(22)])
    result = audit_atari.score_matches('ALE/Pong-v5', [episode(21) for _ in range(20)])
    assert result['mastery_passed']
    assert result['bootstrap_streams'] == 1
    assert result['mean_return_stream_bootstrap_95'] is None
    assert result['natural_win_wilson_95'][0] == pytest.approx(0.8388748419)


def checkpoint_fixture(path):
    metadata = dict(format=3, architecture='dreamerv3-visual-features', config={}, collection_streams=8,
                    perception={'kind': 'levjepa'}, environment_step=200000, learner_step=12405,
                    tensor_sha256={})
    provenance = dict(perception=metadata['perception'], **{key: 'revision' for key in (
        'dreamerv3_revision', 'meganeura_revision', 'blade_revision', 'future_head_revision')})
    metadata.update({key: value for key, value in provenance.items() if key != 'perception'})
    for name in ('world', 'behavior', 'slow_value'):
        file = path / f'{name}.safetensors'
        save_file({'parameter': np.ones((2,), dtype=np.float32),
                   'adam_v.parameter': np.ones((2,), dtype=np.float32)}, file)
        metadata['tensor_sha256'][name] = audit_atari.sha256(file)
    (path / 'metadata.json').write_text(json.dumps(metadata))
    identity = dict(metadata_sha256=audit_atari.sha256(path / 'metadata.json'),
                    tensor_sha256=copy.deepcopy(metadata['tensor_sha256']))
    training = dict(start=dict(config={}, num_envs=8, model_provenance=provenance),
                    end=dict(environment_step=200000, learner_step=12405, run_step=200000),
                    checkpoint=dict(run_step=200000, learner_step=12405, identity=copy.deepcopy(identity)))
    evaluation = dict(start=dict(config={}, model_provenance=provenance, starting_environment_step=200000,
                                starting_learner_step=12405, restored_checkpoint=identity))
    return training, evaluation


def test_checkpoint_uses_nested_perception_and_variable_update_budget(tmp_path):
    training, evaluation = checkpoint_fixture(tmp_path)
    result = audit_atari.verify_checkpoint(tmp_path, training, evaluation, tmp_path)
    assert result['finite_and_complete']
    assert result['tensors'] == dict(world=2, behavior=2, slow_value=2)


@pytest.mark.parametrize('mutation, message', [
    (lambda train, evaluation: train['checkpoint'].update(run_step=100000), 'missing final save'),
    (lambda train, evaluation: evaluation['start'].update(starting_environment_step=100000), 'not the declared final'),
    (lambda train, evaluation: evaluation['start']['restored_checkpoint'].update(metadata_sha256='wrong'), 'different saved/restored'),
    (lambda train, evaluation: evaluation['start'].update(config={'seed': 99}), 'changed checkpoint config'),
    (lambda train, evaluation: train['start'].update(num_envs=4), 'changed training stream count'),
])
def test_checkpoint_identity_failures(tmp_path, mutation, message):
    training, evaluation = checkpoint_fixture(tmp_path)
    mutation(training, evaluation)
    with pytest.raises(ValueError, match=message):
        audit_atari.verify_checkpoint(tmp_path, training, evaluation, tmp_path)


def test_complete_tensor_schema_is_required(tmp_path):
    candidate, reference = tmp_path / 'candidate', tmp_path / 'reference'
    candidate.mkdir()
    reference.mkdir()
    training, evaluation = checkpoint_fixture(candidate)
    checkpoint_fixture(reference)
    save_file({'parameter': np.ones((2,), dtype=np.float32)}, candidate / 'world.safetensors')
    metadata = json.loads((candidate / 'metadata.json').read_text())
    metadata['tensor_sha256']['world'] = audit_atari.sha256(candidate / 'world.safetensors')
    (candidate / 'metadata.json').write_text(json.dumps(metadata))
    identity = dict(metadata_sha256=audit_atari.sha256(candidate / 'metadata.json'),
                    tensor_sha256=metadata['tensor_sha256'])
    training['checkpoint']['identity'] = evaluation['start']['restored_checkpoint'] = identity
    with pytest.raises(ValueError, match='incomplete tensor names'):
        audit_atari.verify_checkpoint(candidate, training, evaluation, reference)


def declaration_fixture(path):
    source = path / 'source'
    examples = source / 'python/examples'
    native = path / 'package/kindle/_native.test.so'
    examples.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    files = [native, examples / 'atari.py', examples / 'atari_vector.py']
    for file in files:
        file.write_text(file.name)
    pins = {str(file): audit_atari.sha256(file) for file in files}
    recipe = dict(batch_size=16, batch_length=64, world_backprop_length=64, world_microbatch_size=16,
                  learning_rate=4e-5, learning_rate_warmup=1000, agc=0.3)
    protocol = dict(environment='ALE/Boxing-v5', protocol='kindle-vector-v2', atari_protocol='published',
                    action_repeat=4, full_action_space=True, sticky_actions=0.0, noop_max=0, max_episode_frames=100000)
    start = dict(**protocol, seed=0, steps=200000, num_envs=8,
                 config=dict(**recipe, seed=0, train_ratio=64, model_size='size12_m',
                             loss_scales=dict(reconstruction=0, future_prediction=0.25)),
                 model_provenance=dict(perception=dict(kind='levjepa')),
                 wrapper_sha256=pins[str(examples / 'atari.py')],
                 runner_sha256=pins[str(examples / 'atari_vector.py')], native_extension_sha256=pins[str(native)],
                 policy_seed_rule='config.seed + stream (wrapping u64)')
    start['environment_seeds'] = [stream * 1_000_003 for stream in range(8)]
    frozen = copy.deepcopy(start)
    frozen.update(seed=100000, steps=75000, mode='evaluate_sample',
                  environment_seeds=[100000 + stream * 1_000_003 for stream in range(8)])
    manifest = dict(**{key: value for key, value in protocol.items() if key != 'protocol'}, **recipe,
                    protocol='kindle-boxing-ratio-pilot-v1', vector_protocol='kindle-vector-v2',
                    training_actions=200000, training_streams=8, training_seeds=[0], ratio_order=[64, 256],
                    frozen_evaluation_actions=75000, frozen_evaluation_streams=8, frozen_environment_seed=100000,
                    frozen_action_mode='sample', frozen_updates=0, model_size='12m', frontend='LeVJEPA',
                    reconstruction=0, future_prediction=0.25, boxing_mastery=audit_atari.MATCH_CRITERIA['ALE/Boxing-v5'],
                    source_worktree=str(source), pins=pins)
    declaration = path / 'manifest.json'
    declaration.write_text(json.dumps(manifest))
    return declaration, dict(start=start), dict(start=frozen, accounting=dict(updates=0))


def test_declared_pilot_is_verified_separately_from_generic_match_scores(tmp_path):
    path, training, evaluation = declaration_fixture(tmp_path)
    result = audit_atari.verify_boxing_declaration(path, training, evaluation)
    assert result['declaration_verified']
    assert result['training_seed'] == 0 and result['train_ratio'] == 64


@pytest.mark.parametrize('mutation, message', [
    (lambda train, evaluation: train['start'].update(steps=100000), 'training budget'),
    (lambda train, evaluation: evaluation['start'].update(steps=20000), 'evaluation budget'),
    (lambda train, evaluation: evaluation['start'].update(mode='evaluate_greedy'), 'evaluation mode'),
    (lambda train, evaluation: evaluation['start'].update(environment_seeds=[0] * 8), 'stream seed rule'),
    (lambda train, evaluation: train['start']['config'].update(agc=0), 'declared recipe'),
    (lambda train, evaluation: train['start']['config'].update(train_ratio=32), 'ratio/model/frontend'),
    (lambda train, evaluation: train['start'].update(native_extension_sha256='wrong'), 'native extension'),
])
def test_changed_pilot_declaration_fails(tmp_path, mutation, message):
    path, training, evaluation = declaration_fixture(tmp_path)
    mutation(training, evaluation)
    with pytest.raises(ValueError, match=message):
        audit_atari.verify_boxing_declaration(path, training, evaluation)


def test_changed_pinned_input_fails(tmp_path):
    path, training, evaluation = declaration_fixture(tmp_path)
    (tmp_path / 'source/python/examples/atari.py').write_text('changed')
    with pytest.raises(ValueError, match='changed pinned campaign input'):
        audit_atari.verify_boxing_declaration(path, training, evaluation)

import hashlib
import random
from pathlib import Path
import sys

import pytest

import kindle
import kindle._exploration as exploration_module
from kindle._exploration import (
    EXPLORATION_KIND, EXPLORATION_PROTOCOL, EXPLORATION_SEED_XOR, PersistentExploration,
)
from kindle._vector_audit import audit
from test_vector import version_two_events, write_log
import atari_vector
import audit_atari
import audit_atari_campaign
from test_atari_campaign import declaration_fixture, run_fixture


def config():
    return dict(kind=EXPLORATION_KIND, probability=0.25, hold_actions=4, seed=73)


@pytest.mark.parametrize('field,value', [
    ('probability', 0), ('probability', -0.1), ('probability', 1.1),
    ('probability', float('nan')), ('probability', float('inf')), ('probability', True),
    ('hold_actions', 0), ('hold_actions', -1), ('hold_actions', 4.0), ('hold_actions', True),
    ('seed', -1), ('seed', 2**64), ('seed', True), ('seed', 1.5), ('kind', 'unknown'),
])
def test_invalid_exploration_configuration(field, value):
    spec = config()
    spec[field] = value
    with pytest.raises(ValueError):
        PersistentExploration(spec, 3, 18)


@pytest.mark.parametrize('spec', [None, {}, {'kind': EXPLORATION_KIND}, dict(config(), extra=True)])
def test_exploration_fields_are_complete_and_versioned(spec):
    with pytest.raises(ValueError):
        PersistentExploration(spec, 3, 18)


@pytest.mark.parametrize('streams,actions', [(0, 18), (True, 18), (2.5, 18), (3, 0), (3, True)])
def test_invalid_streams_or_vocabulary(streams, actions):
    with pytest.raises(ValueError):
        PersistentExploration(config(), streams, actions)


def test_fixed_blocks_match_an_independent_rng_reference():
    spec = config()
    actual = PersistentExploration(spec, 3, 18)
    rngs = [random.Random((73 + stream) ^ EXPLORATION_SEED_XOR) for stream in range(3)]
    counts = [0] * 3
    for _ in range(25):
        expected = [rng.randrange(18) if rng.random() < 0.25 else None for rng in rngs]
        for _ in range(4):
            assert actual.actions() == expected
            counts = [count + (action is not None) for count, action in zip(counts, expected)]
    assert actual.overridden_actions == counts
    spec['probability'] = 1
    assert actual.config['probability'] == 0.25
    actual.config['seed'] = 0
    actual.overridden_actions[0] = -1
    assert actual.config['seed'] == 73 and actual.overridden_actions == counts


def test_resetting_one_stream_preserves_the_others():
    left, right = (PersistentExploration(config(), 3, 18) for _ in range(2))
    for step in range(100):
        if step % 3 == 0:
            left.reset([1])
        a, b = left.actions(), right.actions()
        assert a[0] == b[0] and a[2] == b[2]
    assert left.overridden_actions[::2] == right.overridden_actions[::2]


def test_reset_cancels_the_hold_without_rewinding_rng_or_counts():
    spec = dict(config(), probability=1.0, hold_actions=64)
    actual = PersistentExploration(spec, 1, 18)
    rng = random.Random(73 ^ EXPLORATION_SEED_XOR)
    for _ in range(40):
        rng.random()
        expected = rng.randrange(18)
        assert actual.actions() == [expected]
        actual.reset([0])
    assert actual.overridden_actions == [40]


@pytest.mark.parametrize('streams', [[0, 0], [0, 3], [-1], [True], [1.0], [None]])
def test_invalid_reset_does_not_partially_mutate_stream_state(streams):
    left, right = (PersistentExploration(config(), 3, 18) for _ in range(2))
    assert left.actions() == right.actions()
    with pytest.raises(ValueError):
        left.reset(streams)
    for _ in range(12):
        assert left.actions() == right.actions()


def test_probability_applies_to_both_policy_and_random_blocks():
    actual = PersistentExploration(dict(config(), hold_actions=16), 8, 18)
    for _ in range(4096):
        actual.actions()
    fraction = sum(actual.overridden_actions) / (8 * 4096)
    assert 0.20 < fraction < 0.30


def version_three_events(probability=1.0, hold_actions=4):
    rows = version_two_events()
    rows[0].update(protocol=EXPLORATION_PROTOCOL, seed=73,
        exploration=dict(config(), probability=probability, hold_actions=hold_actions),
        exploration_sha256=hashlib.sha256(Path(exploration_module.__file__).read_bytes()).hexdigest())
    rows[0]['config']['seed'] = 73
    explorer = PersistentExploration(rows[0]['exploration'], 2, 3)
    for row in rows[1:]:
        if row['event'] == 'transition':
            row['action_overrides'] = explorer.actions()
            row['actions'] = [action if forced is None else forced
                              for action, forced in zip(row['actions'], row['action_overrides'])]
        elif row['event'] == 'reset':
            explorer.reset(row['streams'])
        elif row['event'] in ('progress', 'run_end'):
            row['overridden_actions'] = explorer.overridden_actions
    return rows


def test_exploration_ledger_keeps_original_action_reward_and_update_accounting(tmp_path):
    result = audit(write_log(tmp_path, version_three_events()))
    assert result['actions'] == 6 and result['updates'] == 3
    assert result['exploration_ledger_verified'] is True
    assert result['overridden_actions'] == [3, 3]
    assert result['natural_episodes'] == 1
    assert result['exploration_sha256'] == hashlib.sha256(
        Path(exploration_module.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize('probability,hold', [(0.25, 4), (1.0, 1), (0.5, 2)])
def test_mixed_policy_blocks_and_resets_keep_the_same_ledger(tmp_path, probability, hold):
    rows = version_three_events(probability, hold)
    result = audit(write_log(tmp_path, rows))
    assert result['actions'] == 6 and result['updates'] == 3
    assert result['overridden_actions'] == rows[-1]['overridden_actions']


@pytest.mark.parametrize('mutation', [
    'header_missing', 'header_unknown', 'helper_hash', 'seed', 'frozen',
    'missing_override', 'wrong_override', 'override_bool', 'executed_action',
    'missing_count', 'wrong_count', 'boolean_count',
])
def test_exploration_audit_rejects_tampered_provenance_and_actions(tmp_path, mutation):
    rows = version_three_events()
    transition = next(row for row in rows if row['event'] == 'transition')
    if mutation == 'header_missing':
        rows[0].pop('exploration')
    elif mutation == 'header_unknown':
        rows[0]['exploration']['kind'] = 'unknown'
    elif mutation == 'helper_hash':
        rows[0]['exploration_sha256'] = 'different'
    elif mutation == 'seed':
        rows[0]['exploration']['seed'] += 1
    elif mutation == 'frozen':
        rows[0]['mode'] = 'evaluate_sample'
    elif mutation == 'missing_override':
        transition.pop('action_overrides')
    elif mutation == 'wrong_override':
        transition['action_overrides'][0] = None
    elif mutation == 'override_bool':
        transition['action_overrides'][0] = True
    elif mutation == 'executed_action':
        transition['actions'][0] = (transition['actions'][0] + 1) % 3
    elif mutation == 'missing_count':
        rows[-1].pop('overridden_actions')
    elif mutation == 'wrong_count':
        rows[-1]['overridden_actions'][0] = 0
    else:
        rows[-1]['overridden_actions'][0] = True
    with pytest.raises((ValueError, KeyError)):
        audit(write_log(tmp_path, rows))


@pytest.mark.parametrize('field', ['header', 'transition', 'count'])
def test_legacy_protocol_cannot_silently_admit_overrides(tmp_path, field):
    rows = version_two_events()
    if field == 'header':
        rows[0]['exploration'] = config()
    elif field == 'transition':
        rows[1]['action_overrides'] = [None, None]
    else:
        rows[-1]['overridden_actions'] = [0, 0]
    with pytest.raises(ValueError, match='undeclared'):
        audit(write_log(tmp_path, rows))


@pytest.mark.parametrize('args,message', [
    (['--exploration-probability', 'nan'], 'probability'),
    (['--exploration-probability', '-0.1'], 'probability'),
    (['--exploration-probability', '1.1'], 'probability'),
    (['--exploration-hold', '0'], 'hold'),
    (['--evaluate', '--exploration-probability', '0.1'], 'frozen evaluation'),
    (['--restore', 'unused', '--exploration-probability', '0.1'], 'fresh run'),
])
def test_runner_rejects_invalid_exploration_before_gpu(tmp_path, monkeypatch, capsys, args, message):
    monkeypatch.setattr(sys, 'argv', ['atari_vector', 'unused', '--output', str(tmp_path / 'run.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / 'run.jsonl').exists()


def test_runner_rejects_an_old_native_api_before_constructing_anything(tmp_path, monkeypatch, capsys):
    class OldAgent:
        def __init__(self, *args, **kwargs):
            pytest.fail('native construction before API validation')

        def act(self, greedy=False):
            pytest.fail('native execution before API validation')

    monkeypatch.setattr(kindle, 'VectorAgent', OldAgent)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *args, **kwargs: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', ['atari_vector', 'unused', '--output', str(tmp_path / 'run.jsonl'),
                                     '--exploration-probability', '0.25'])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires native vector action overrides' in capsys.readouterr().err
    assert not (tmp_path / 'run.jsonl').exists()


def test_final_pair_allows_exploration_only_during_training(tmp_path):
    declaration = declaration_fixture(tmp_path)
    row = declaration['runs'][0]
    training, evaluation = run_fixture(declaration, row)
    audit_atari.verify_final_pair(training, evaluation)
    training['start']['protocol'] = EXPLORATION_PROTOCOL
    audit_atari.verify_final_pair(training, evaluation)
    # A task score is not permission to change an already declared replication.
    with pytest.raises(ValueError, match='changed declared header: protocol'):
        audit_atari_campaign.verify_run_declaration(declaration, row, training, evaluation)
    evaluation['start']['protocol'] = EXPLORATION_PROTOCOL
    with pytest.raises(ValueError, match='changed evaluation identity: protocol'):
        audit_atari.verify_final_pair(training, evaluation)


@pytest.mark.parametrize('field', ['config', 'native_extension_sha256', 'runner_sha256',
                                  'action_repeat', 'model_provenance', 'sticky_actions'])
def test_exploration_does_not_relax_final_model_or_game_identity(tmp_path, field):
    declaration = declaration_fixture(tmp_path)
    training, evaluation = run_fixture(declaration, declaration['runs'][0])
    training['start']['protocol'] = EXPLORATION_PROTOCOL
    evaluation['start'][field] = 'changed'
    with pytest.raises(ValueError, match='changed evaluation identity: ' + field):
        audit_atari.verify_final_pair(training, evaluation)

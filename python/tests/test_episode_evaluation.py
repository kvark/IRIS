"""Fabricated frozen vector streams; no GPU, policy-quality, or speed claims."""

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import kindle
from kindle._vector_audit import EPISODE_EVALUATION_PROTOCOL, VECTOR_PROTOCOL, audit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import atari_vector
import audit_atari
import audit_atari_campaign
from test_atari_campaign import declaration_fixture, run_fixture


@pytest.fixture
def frozen_run(monkeypatch, tmp_path):
    created = []
    interrupt = False

    class Environment:
        action_space = SimpleNamespace(n=2)
        action_meanings = ['NOOP', 'FIRE']
        executed_action_frames = reset_noop_frames = emulator_resets = 0
        closed = False

        def __init__(self, stream):
            self.stream = stream

        def reset(self, *, seed=None):
            self.length = 0
            self.emulator_resets += 1
            return None, {}

        def step(self, action):
            assert action == 0
            self.length += 1
            self.executed_action_frames += 4
            ended = self.length == 2 + self.stream
            return None, -1.0 - self.stream, ended and self.stream == 0, ended and self.stream == 1, {}

        def close(self):
            self.closed = True

    class Agent:
        environment_step, learner_step = 100, 7
        replay_len = 0
        training_debt = 0.0
        provenance = {'fixture': True}
        gpu_device = {'fixture': True}
        cpu_worker_threads = 1
        trainable_parameter_counts = {'world': 0, 'behavior': 0}

        @classmethod
        def restore(cls, checkpoint, encoder, streams):
            instance = cls()
            instance.streams = streams
            instance.config = kindle.default_config(2)
            return instance

        def begin_episodes(self, ids, frames):
            self.replay_len += len(ids)

        def act(self, *, greedy):
            if interrupt:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return [0] * self.streams

        def observe(self, ids, frames, rewards, terminated, truncated):
            self.environment_step += len(ids)
            self.replay_len += len(ids)
            return [[reward, 0.] for reward in rewards]

        def learn_scheduled(self):
            raise AssertionError('frozen evaluation learned')

        def save_checkpoint(self, path):
            raise AssertionError('frozen evaluation wrote a checkpoint')

    def make(*_, **__):
        env = Environment(len(created))
        created.append(env)
        return env

    monkeypatch.setattr(atari_vector.gym, 'make', make)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    monkeypatch.setattr(atari_vector, 'checkpoint_identity', lambda path: {'fixture': True})
    monkeypatch.setattr(kindle, 'VectorAgent', Agent)

    def run(target=2, cap=100, interrupted=False):
        nonlocal interrupt
        interrupt = interrupted
        output = tmp_path / f'vector-{target}-{cap}.jsonl'
        args = ['atari_vector.py', 'unused', '--output', str(output), '--num-envs', '2',
                '--steps', str(cap), '--evaluate', '--restore', 'fixture', '--report-every', '2']
        if target is not None:
            args += ['--episodes-per-env', str(target)]
        monkeypatch.setattr(sys, 'argv', args)
        atari_vector.main()
        assert len(created) == 2 and all(env.closed for env in created)
        return output, [json.loads(line) for line in output.read_text().splitlines()]

    return run


@pytest.mark.parametrize('target, cap, expected_actions, expected_counts, reason', [
    (1, 100, 6, [1, 1], 'episode_budget_complete'),
    (2, 100, 12, [3, 2], 'episode_budget_complete'),
    (2, 12, 12, [3, 2], 'episode_budget_complete'),
    (2, 10, 10, [2, 1], 'action_cap_reached'),
    (None, 14, 14, [3, 2], 'budget_complete'),
])
def test_runner_stops_at_first_settled_target_or_cap(frozen_run, target, cap, expected_actions, expected_counts, reason):
    path, rows = frozen_run(target, cap)
    result = audit(path)
    assert rows[0]['protocol'] == (EPISODE_EVALUATION_PROTOCOL if target else VECTOR_PROTOCOL)
    assert rows[-1]['reason'] == reason and rows[-1]['episode_counts'] == expected_counts
    assert result['actions'] == expected_actions and result['updates'] == 0
    assert result['completed_episodes'] == sum(expected_counts)
    assert result['natural_episodes'] == expected_counts[0]
    assert result['truncated_episodes'] == expected_counts[1]
    assert result['positive_return_natural_episodes'] == 0
    assert result['budget_complete'] == (reason != 'action_cap_reached')
    assert len([row for row in rows if row['event'] == 'episode']) == sum(expected_counts)
    if target:
        assert result['episode_budget_complete'] == result['budget_complete']
        assert result['action_cap_reached'] == (expected_actions == cap)
        assert result['evaluation_episodes_per_stream'] == target
    else:
        assert 'evaluation_episodes_per_stream' not in rows[0]
        assert 'episode_budget_complete' not in result
    if result['budget_complete']:
        assert audit_atari.read_run(path)['accounting'] == result
    else:
        with pytest.raises(ValueError, match='incomplete declared run budget'):
            audit_atari.read_run(path)


def test_interrupt_does_not_complete_episode_budget(frozen_run):
    path, rows = frozen_run(interrupted=True)
    result = audit(path)
    assert rows[-1]['reason'] == 'interrupted'
    assert result['actions'] == 2 and not result['budget_complete']


@pytest.mark.parametrize('args', [
    ['--episodes-per-env', '0', '--evaluate', '--restore', 'unused'],
    ['--episodes-per-env', '-1', '--evaluate', '--restore', 'unused'],
    ['--episodes-per-env', '1'],
    ['--episodes-per-env', '1', '--evaluate'],
    ['--episodes-per-env', '1', '--restore', 'unused'],
    ['--episodes-per-env', '1', '--evaluate', '--restore', 'unused', '--checkpoint', 'unused'],
])
def test_cli_rejects_ambiguous_episode_budget_before_gpu(monkeypatch, tmp_path, args):
    output = tmp_path / 'never-created.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'unused', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize('mutation', [
    lambda rows: rows[0].update(evaluation_episodes_per_stream=0),
    lambda rows: rows[0].update(evaluation_episodes_per_stream=True),
    lambda rows: rows[0].update(evaluation_episodes_per_stream='2'),
    lambda rows: rows[0].update(evaluation_episodes_per_stream=1),
    lambda rows: rows[0].update(mode='train'),
    lambda rows: rows[0].update(restored_checkpoint=None),
    lambda rows: rows[0].update(protocol=VECTOR_PROTOCOL),
    lambda rows: rows[-1].update(reason='budget_complete'),
    lambda rows: rows[-1].update(reason='action_cap_reached'),
    lambda rows: rows[-1].update(episode_counts=[2, 2]),
    lambda rows: rows[0].update(steps=10),
    lambda rows: rows[0].update(steps=True),
    lambda rows: rows.insert(-1, dict(event='checkpoint')),
])
def test_auditor_rejects_hidden_late_incomplete_or_training_stops(frozen_run, mutation):
    path, rows = frozen_run()
    mutation(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        audit(path)


def test_action_cap_without_all_stream_targets_cannot_claim_completion(frozen_run):
    path, rows = frozen_run(cap=10)
    rows[-1]['reason'] = 'episode_budget_complete'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='wrong episode-budget stop reason'):
        audit(path)


def test_final_pair_support_does_not_rewrite_existing_campaigns(tmp_path):
    declaration = declaration_fixture(tmp_path)
    row = declaration['runs'][0]
    training, evaluation = run_fixture(declaration, row)
    evaluation['start'].update(protocol=EPISODE_EVALUATION_PROTOCOL, evaluation_episodes_per_stream=4)
    audit_atari.verify_final_pair(training, evaluation)
    with pytest.raises(ValueError, match='changed declared header: protocol'):
        audit_atari_campaign.verify_run_declaration(declaration, row, training, evaluation)
    training['start']['protocol'] = EPISODE_EVALUATION_PROTOCOL
    with pytest.raises(ValueError, match='frozen only'):
        audit_atari.verify_final_pair(training, evaluation)


@pytest.mark.parametrize('field', ['config', 'native_extension_sha256', 'runner_sha256',
                                  'action_repeat', 'model_provenance', 'sticky_actions'])
def test_episode_evaluation_preserves_model_and_game_identity(tmp_path, field):
    declaration = declaration_fixture(tmp_path)
    training, evaluation = run_fixture(declaration, declaration['runs'][0])
    evaluation['start'].update(protocol=EPISODE_EVALUATION_PROTOCOL, evaluation_episodes_per_stream=4)
    evaluation['start'][field] = 'changed'
    with pytest.raises(ValueError, match='changed evaluation identity: ' + field):
        audit_atari.verify_final_pair(training, evaluation)

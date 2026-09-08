import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from atari_tasks import EpisodeTask, ROM_SHA256
from replay_atari import replay_rows


def replay_fixture():
    class Environment:
        def __init__(self, stream):
            self.stream = stream
            self.executed_action_frames = 0
            self.monitor = SimpleNamespace()
            self.reset()

        def reset(self):
            self.steps = 0
            self.monitor.task = EpisodeTask('ALE/Boxing-v5', ROM_SHA256['ALE/Boxing-v5'])

        def step(self, action):
            assert action == self.stream
            self.steps += 1
            self.executed_action_frames += 4
            reward = 1.0 if self.stream == 0 else -1.0
            for _ in range(3):
                self.monitor.task.advance(0)
            self.monitor.task.advance(reward)
            return None, reward, self.stream == 0 and self.steps == 2, False, {}

    rows = []
    for tick in range(1, 4):
        rows.append(dict(event='transition', run_step=tick * 2, actions=[0, 1], rewards=[1.0, -1.0],
                         terminated=[tick == 2, False], truncated=[False, False],
                         executed_action_frames=[tick * 4, tick * 4]))
        if tick == 2:
            rows.append(dict(event='episode', stream=0, episode_return=2.0, terminated=True, truncated=False))
            rows.append(dict(event='reset', streams=[0]))
    environments = [Environment(0), Environment(1)]
    return rows, environments, [environment.monitor for environment in environments]


def source(rows):
    return io.StringIO(''.join(json.dumps(row) + '\n' for row in rows))


def test_replay_preserves_independent_episodes_and_partial_tails():
    rows, environments, monitors = replay_fixture()
    episodes, partial = replay_rows(source(rows), environments, monitors)
    assert len(episodes) == 1
    assert episodes[0]['task_outcome']['episode_success']
    assert episodes[0]['first_frame'] == 0 and episodes[0]['last_frame'] == 8
    assert [row['episode_frames'] for row in partial] == [4, 12]
    assert [row['episode_score'] for row in partial] == [1, -3]
    assert not any(row['eligible_completed_episode'] for row in partial)


@pytest.mark.parametrize('key, value', [
    ('rewards', [2, -1]), ('terminated', [True, False]), ('truncated', [True, False]),
    ('executed_action_frames', [3, 4]),
])
def test_replay_rejects_every_transition_mismatch(key, value):
    rows, environments, monitors = replay_fixture()
    rows[0][key] = value
    with pytest.raises(ValueError, match=f'replay diverged.*{key}'):
        replay_rows(source(rows), environments, monitors)


def test_replay_rejects_a_changed_episode_score():
    rows, environments, monitors = replay_fixture()
    next(row for row in rows if row['event'] == 'episode')['episode_return'] = 3
    with pytest.raises(ValueError, match='task score differs'):
        replay_rows(source(rows), environments, monitors)


def test_replay_rejects_learning_records():
    rows, environments, monitors = replay_fixture()
    rows.insert(0, dict(event='learner'))
    with pytest.raises(ValueError, match='source contains learning'):
        replay_rows(source(rows), environments, monitors)

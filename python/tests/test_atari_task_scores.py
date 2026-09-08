import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import audit_atari_tasks as scorer


def outcome(game, score, *, pyramid=True, terminal=True, truncated=False, frames=100):
    environment = f'ALE/{game}-v5'
    milestone = pyramid if game == 'Qbert' else score >= (25 if game == 'Freeway' else 864)
    completed = terminal or truncated
    return dict(task=scorer.TASKS[environment], episode_score=score, episode_frames=frames,
                first_milestone_frame=min(50, frames) if milestone else None,
                max_initial_qbert_cubes=(21 if pyramid else 0) if game == 'Qbert' else None,
                terminated=terminal, truncated=truncated, eligible_completed_episode=completed,
                episode_success=completed and milestone and (game != 'Freeway' or terminal and not truncated))


def episodes(game, scores, **kwargs):
    return [dict(stream=index % 2, episode=index // 2, episode_return=score,
                 terminated=kwargs.get('terminal', True), truncated=kwargs.get('truncated', False),
                 task_outcome=outcome(game, score, **kwargs)) for index, score in enumerate(scores)]


def score(game, scores, **kwargs):
    return scorer.score_tasks(f'ALE/{game}-v5', episodes(game, scores, **kwargs))


@pytest.mark.parametrize('game, returns, passed', [
    ('Freeway', [25] * 20, True),
    ('Freeway', [30] * 19, False),
    ('Freeway', [24] * 20, False),
    ('Freeway', [30] * 18 + [0] * 2, True),
    ('Freeway', [25] * 18 + [0] * 2, False),
    ('Freeway', [30] * 17 + [0] * 3, False),
    ('Breakout', [864] * 20, True),
    ('Breakout', [864] * 19, False),
    ('Breakout', [432] * 20, False),
    ('Breakout', [864] * 18 + [0] * 2, True),
    ('Breakout', [864] * 17 + [0] * 3, False),
    ('Qbert', [15000] * 20, True),
    ('Qbert', [3950] * 20, False),
    ('Qbert', [15000] * 19, False),
    ('Qbert', [14999] * 20, False),
])
def test_each_predeclared_task_gate(game, returns, passed):
    result = score(game, returns)
    assert result['task_gate_passed'] is passed
    assert result['completed_episodes'] == len(returns)
    assert not result['reliability_assessed']


def test_qbert_high_score_cannot_replace_actual_pyramid_completion():
    result = score('Qbert', [20000] * 20, pyramid=False)
    assert result['mean_completed_return'] == 20000 and result['task_successes'] == 0
    assert not result['task_gate_passed']
    rows = episodes('Qbert', [15000] * 20)
    for row in rows[-3:]:
        row['task_outcome'] = outcome('Qbert', 15000, pyramid=False)
    assert not scorer.score_tasks('ALE/Qbert-v5', rows)['task_gate_passed']


@pytest.mark.parametrize('game, value', [('Breakout', 864), ('Qbert', 15000)])
def test_clear_before_cutoff_is_retained_without_claiming_natural_wins(game, value):
    result = score(game, [value] * 20, terminal=False, truncated=True)
    assert result['task_gate_passed'] and result['task_successes'] == 20
    assert result['natural_episodes'] == 0 and result['truncated_episodes'] == 20


def test_freeway_timeouts_never_count_as_complete_round_success():
    result = score('Freeway', [30] * 20, terminal=False, truncated=True)
    assert result['task_successes'] == 0 and not result['task_gate_passed']
    rows = episodes('Freeway', [30] * 21)
    rows[-1].update(terminated=False, truncated=True, task_outcome=outcome('Freeway', 30, terminal=False, truncated=True))
    result = scorer.score_tasks('ALE/Freeway-v5', rows)
    assert result['task_success_fraction'] > 0.9 and result['mean_completed_return'] == 30
    assert not result['task_gate_passed']


@pytest.mark.parametrize('game', ['Freeway', 'Breakout', 'Qbert'])
def test_empty_sample_and_partial_only_are_not_competence(game):
    result = score(game, [])
    assert not result['task_gate_passed']
    assert result['mean_completed_return'] is result['task_success_fraction'] is None
    with pytest.raises(ValueError, match='boundary'):
        score(game, [864] * 20, terminal=False, truncated=False)


def test_duplicate_completed_episode_is_rejected():
    row = episodes('Breakout', [864])[0]
    with pytest.raises(ValueError, match='duplicate'):
        scorer.score_tasks('ALE/Breakout-v5', [row] * 20)


@pytest.mark.parametrize('mutation, message', [
    (lambda task: task.update(episode_score=True), 'task score'),
    (lambda task: task.update(episode_score=-1), 'task score'),
    (lambda task: task.update(episode_score=float('nan')), 'task score'),
    (lambda task: task.update(episode_score=25.5), 'task score'),
    (lambda task: task.update(episode_frames=True), 'frame count'),
    (lambda task: task.update(episode_frames=0), 'frame count'),
    (lambda task: task.update(first_milestone_frame=0), 'milestone frame'),
    (lambda task: task.update(first_milestone_frame=101), 'milestone frame'),
    (lambda task: task.update(first_milestone_frame=None), 'score and milestone'),
    (lambda task: task.update(terminated=1), 'task boundary'),
    (lambda task: task.update(eligible_completed_episode=1), 'task eligibility'),
    (lambda task: task.update(episode_success=1), 'task-success'),
    (lambda task: task.update(max_initial_qbert_cubes=21), 'Qbert evidence'),
    (lambda task: task.update(task='first_pyramid_all_21_cubes'), 'wrong task'),
])
def test_malformed_task_evidence_is_rejected(mutation, message):
    rows = episodes('Freeway', [25])
    mutation(rows[0]['task_outcome'])
    with pytest.raises(ValueError, match=message):
        scorer.score_tasks('ALE/Freeway-v5', rows)


def test_inconsistent_qbert_and_impossible_breakout_scores_fail():
    task = outcome('Qbert', 15000)
    task['max_initial_qbert_cubes'] = 20
    with pytest.raises(ValueError, match='pyramid evidence'):
        scorer.check_task('ALE/Qbert-v5', task, True)
    with pytest.raises(ValueError, match='Breakout final score'):
        score('Breakout', [865] * 20)
    with pytest.raises(ValueError, match='no supported task'):
        scorer.score_tasks('ALE/Pong-v5', [])


def replay_fixture(tmp_path):
    environment = 'ALE/Qbert-v5'
    wrapper = scorer.sha256(scorer.replay_atari.atari.__file__)
    native = Path(scorer.ale_native.__file__)
    rom = scorer.rom_identity(environment)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(environment=environment,
        pins={rom['path']: rom['sha256'], str(native): scorer.sha256(native)})))
    header = dict(environment=environment, num_envs=2, mode='evaluate_sample',
                  ale_py_version='0.12.1', wrapper_sha256=wrapper, atari_protocol='published',
                  action_repeat=4, sticky_actions=0, noop_max=0, full_action_space=True, max_episode_frames=100000)
    path = tmp_path / 'synthetic-evaluation-binding.txt'
    path.write_text('Unit fixture for replay binding, not an executable evaluation log.\n')
    rows = episodes('Qbert', [3950] * 20)
    recorded = [{key: value for key, value in row.items() if key != 'task_outcome'} for row in rows]
    for row in rows:
        row['first_frame'] = row['episode'] * 100
        row['last_frame'] = row['first_frame'] + 100
    partial = [dict(stream=0, **outcome('Qbert', 0, pyramid=False, terminal=False, frames=0)),
               dict(stream=1, **outcome('Qbert', 50000, terminal=False, frames=50))]
    evaluation = dict(path=str(path), sha256=scorer.sha256(path), start=header,
                      accounting=dict(updates=0, budget_complete=True), episodes=recorded,
                      end=dict(partial_returns=[0, 50000], executed_action_frames=[1000, 1050]))
    replay = dict(protocol='kindle-atari-task-replay-v1', source_log=str(path),
                  source_log_sha256=evaluation['sha256'], source_header=copy.deepcopy(header),
                  source_accounting=copy.deepcopy(evaluation['accounting']), agent_constructed=False,
                  learner_updates=0, full_trajectory_replayed=True, source_manifest=str(manifest),
                  source_manifest_sha256=scorer.sha256(manifest), rom=rom,
                  ale_native_sha256=scorer.sha256(native), wrapper_sha256=wrapper,
                  replay_script_sha256=scorer.sha256(scorer.replay_atari.__file__),
                  observer_sha256=scorer.sha256(scorer.atari_tasks.__file__), episodes=rows,
                  partial=partial, task_successes=20)
    return evaluation, replay


def test_replay_bindings_and_partial_frames_are_verified(tmp_path):
    evaluation, replay = replay_fixture(tmp_path)
    scorer.check_replay(evaluation, replay)
    result = scorer.score_tasks(evaluation['start']['environment'], replay['episodes'])
    assert result['task_successes'] == 20
    assert result['mean_completed_return'] == 3950 and not result['task_gate_passed']
    assert replay['partial'][1]['episode_score'] == 50000


@pytest.mark.parametrize('mutation, message', [
    (lambda replay: replay.update(source_log_sha256='wrong'), 'replay source differs'),
    (lambda replay: replay['source_header'].update(mode='train'), 'replay source differs'),
    (lambda replay: replay.update(agent_constructed=True), 'complete CPU replay'),
    (lambda replay: replay.update(learner_updates=1), 'complete CPU replay'),
    (lambda replay: replay.update(full_trajectory_replayed=False), 'complete CPU replay'),
    (lambda replay: replay.update(source_manifest_sha256='wrong'), 'manifest changed'),
    (lambda replay: replay['rom'].update(sha256='wrong'), 'ROM differs'),
    (lambda replay: replay.update(observer_sha256='wrong'), 'implementation changed'),
    (lambda replay: replay.update(replay_script_sha256='wrong'), 'implementation changed'),
    (lambda replay: replay['episodes'].pop(), 'episode count differs'),
    (lambda replay: replay['episodes'][0].update(episode_return=50000), 'episode ledger differs'),
    (lambda replay: replay['episodes'][2].update(first_frame=0), 'episode frames differ'),
    (lambda replay: replay['partial'].reverse(), 'partial streams'),
    (lambda replay: replay['partial'][1].update(episode_frames=49), 'milestone frame'),
    (lambda replay: replay['partial'][1].update(episode_score=50001), 'partial task ledger'),
    (lambda replay: replay.update(task_successes=21), 'success count differs'),
])
def test_replay_mismatches_fail(tmp_path, mutation, message):
    evaluation, replay = replay_fixture(tmp_path)
    mutation(replay)
    with pytest.raises(ValueError, match=message):
        scorer.check_replay(evaluation, replay)


@pytest.mark.parametrize('exploration', [False, True])
def test_final_audit_calls_complete_checkpoint_guard_and_does_not_certify_campaign(tmp_path, monkeypatch, exploration):
    evaluation, replay = replay_fixture(tmp_path)
    header = evaluation['start']
    for key in ('protocol', 'action_meanings', 'config', 'model_provenance', 'native_extension_sha256',
                'runner_sha256', 'trainable_parameter_counts', 'gpu_device', 'cpu_worker_threads'):
        header[key] = 'unit fixture'
    replay['source_header'] = copy.deepcopy(header)
    training = copy.deepcopy(evaluation)
    training['start'].update(mode='train', starting_environment_step=0,
                             starting_learner_step=0, restored_checkpoint=None)
    if exploration:
        training['start']['protocol'] = 'kindle-vector-v3'
        evaluation['start']['protocol'] = 'kindle-vector-v2'
        replay['source_header'] = copy.deepcopy(evaluation['start'])
    training['accounting']['updates'] = 1
    replay_path = tmp_path / 'replay.json'
    replay_path.write_text(json.dumps(replay))
    monkeypatch.setattr(scorer, 'read_run', lambda path: training if path == 'train' else evaluation)
    calls = []

    def checkpoint_guard(path, train, frozen, schema):
        calls.append((path, train, frozen, schema))
        return {'unit_fixture': True}

    monkeypatch.setattr(scorer, 'verify_checkpoint', checkpoint_guard)
    result = scorer.audit_final_tasks('train', 'evaluate', 'checkpoint', 'schema', replay_path)
    assert calls == [('checkpoint', training, evaluation, 'schema')]
    assert not result['campaign_declaration_verified'] and not result['reliability_assessed']
    assert not result['task_gate_passed']
    training['start']['sticky_actions'] = 0.25
    with pytest.raises(ValueError, match='changed evaluation identity: sticky_actions'):
        scorer.audit_final_tasks('train', 'evaluate', 'checkpoint', 'schema', replay_path)


@pytest.mark.parametrize('mutation, message', [
    (lambda train, frozen: train['start'].update(mode='evaluate_sample'), 'train/evaluation mode'),
    (lambda train, frozen: frozen['start'].update(mode='evaluate_greedy'), 'train/evaluation mode'),
    (lambda train, frozen: train['start'].update(starting_environment_step=1), 'fresh and have updates'),
    (lambda train, frozen: train['start'].update(starting_learner_step=1), 'fresh and have updates'),
    (lambda train, frozen: train['start'].update(restored_checkpoint={}), 'fresh and have updates'),
    (lambda train, frozen: train['accounting'].update(updates=0), 'fresh and have updates'),
    (lambda train, frozen: frozen['accounting'].update(updates=1), 'evaluation is not frozen'),
])
def test_final_audit_rejects_wrong_learning_lifecycle(monkeypatch, mutation, message):
    training = dict(start=dict(mode='train', starting_environment_step=0,
                              starting_learner_step=0, restored_checkpoint=None), accounting=dict(updates=1))
    evaluation = dict(start=dict(mode='evaluate_sample'), accounting=dict(updates=0))
    mutation(training, evaluation)
    monkeypatch.setattr(scorer, 'read_run', lambda path: training if path == 'train' else evaluation)
    with pytest.raises(ValueError, match=message):
        scorer.audit_final_tasks('train', 'evaluate', 'checkpoint', 'schema', 'not-read')

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import audit_atari_campaign as campaign


def runtime_fixture(tmp_path, config, pins, streams):
    """Synthetic gate data for validation tests, never hardware evidence."""
    manifest = dict(protocol='kindle-vector-memory-runtime-v1', order=[8, 6, 4, 4, 6, 8],
                    required_free_mib=2048, config=dict(config, seed=7301), pins=dict(pins),
                    utc='1970-01-01T00:13:20+00:00', training_actions=3840, frozen_actions=768)
    rows = []
    for index in range(2):
        start = 840 + index * 40
        rows.append(dict(name=f'order{index}-n{streams}', streams=streams,
                         state=dict(metadata=dict(config=manifest['config'])),
                         train_trace_sha256='unit-training-trace', restore_trace_sha256='unit-restore-trace',
                         checkpoint=dict(finite_and_complete=True),
                         training=dict(accounting=dict(actions=3840)),
                         restore=dict(accounting=dict(actions=768, updates=0)),
                         **{phase + '_phase': dict(command_start=start, command_end=start + 10)
                            for phase in ('train', 'restore')},
                         **{phase + '_gpu': dict(coverage_passed=True, free_memory_gate_passed=True,
                                                minimum_reported_free_mib=2300)
                            for phase in ('train', 'restore')}))
    result = dict(protocol=manifest['protocol'], runs=rows,
                  pairs=[dict(streams=streams, repeated_state_and_traces_exact=True, memory_gate_passed=True)])
    paths = {name: tmp_path / f'runtime-{name}.json' for name in ('manifest', 'summary')}
    for name, value in (('manifest', manifest), ('summary', result)):
        paths[name].write_text(json.dumps(value))
        pins[str(paths[name])] = campaign.sha256(paths[name])
    return {name: str(path) for name, path in paths.items()}


def declaration_fixture(tmp_path, streams=8):
    schema = tmp_path / 'schema'
    schema.mkdir()
    fixtures = [schema / name for name in
                ('metadata.json', 'world.safetensors', 'behavior.safetensors', 'slow_value.safetensors')]
    fixtures += [tmp_path / 'native-unit-fixture', tmp_path / 'encoder-unit-fixture']
    for path in fixtures:
        path.write_text('Unit input pin, not a model: ' + path.name)
    modules = (campaign, campaign.matches, campaign.tasks, campaign.replay_atari,
               campaign.atari_tasks, campaign.check_atari_adapter, campaign.vector_audit,
               campaign.ale_native, campaign.replay_atari.atari)
    paths = fixtures + [Path(module.__file__).resolve() for module in modules]
    runner = Path(campaign.__file__).with_name('atari_vector.py')
    paths += [runner]
    for game in campaign.CRITERIA:
        paths.append(Path(campaign.check_atari_adapter.rom_identity(game)['path']))
    pins = {str(path): campaign.sha256(path) for path in paths}
    inputs = dict(runner=str(runner), wrapper=str(Path(campaign.replay_atari.atari.__file__).resolve()),
                  native=str(fixtures[-2]), encoder=str(fixtures[-1]))
    header = dict(campaign.ATARI_HEADER, action_meanings=[str(index) for index in range(18)],
                  model_provenance=dict(perception=dict(kind='levjepa', checkpoint_sha256=pins[inputs['encoder']])),
                  trainable_parameter_counts=[100, 10], gpu_device={'name': 'unit fixture'}, cpu_worker_threads=8)
    for key, role in (('runner_sha256', 'runner'), ('wrapper_sha256', 'wrapper'), ('native_extension_sha256', 'native')):
        header[key] = pins[inputs[role]]
    rows = []
    for game in campaign.CRITERIA:
        for seed in campaign.SEEDS:
            name = game.split('/')[1] + '-' + str(seed)
            rows.append(dict(environment=game, seed=seed,
                             **{key: str(tmp_path / (name + '-' + key))
                                for key in ('training', 'evaluation', 'checkpoint', 'replay')}))
    config = dict(action_count=18, extrinsic_reward_scale=1, intrinsic_reward_scale=0,
                  visitation_bonus=False, batch_size=16, batch_length=64, train_ratio=64,
                  loss_scales=dict(reconstruction=0, future_prediction=0.25))
    gate = runtime_fixture(tmp_path, config, pins, streams)
    return dict(protocol=campaign.PROTOCOL, training_seeds=campaign.SEEDS.copy(), runtime_gate=gate,
                declared_unix_time=1000.0, training_streams=streams, evaluation_streams=streams, evaluation_seed=100000,
                games={game: dict(training_actions=200016, evaluation_actions=400032, criteria=copy.deepcopy(criteria))
                       for game, criteria in campaign.CRITERIA.items()},
                config=config, header=header, pins=pins, inputs=inputs, schema=str(schema), runs=rows)


def run_fixture(declaration, row):
    game = declaration['games'][row['environment']]
    streams = declaration['training_streams']
    header = dict(copy.deepcopy(declaration['header']), environment=row['environment'],
                  num_envs=streams, config=dict(copy.deepcopy(declaration['config']), seed=row['seed']))
    training = dict(path=row['training'], sha256=row['training'], start=dict(copy.deepcopy(header),
                    mode='train', seed=row['seed'], steps=game['training_actions'], unix_time=2000.0,
                    environment_seeds=[row['seed'] + stream * 1_000_003 for stream in range(streams)],
                    restored_checkpoint=None, starting_environment_step=0, starting_learner_step=0),
                    accounting=dict(actions=game['training_actions'], budget_complete=True, updates=12405),
                    end=dict(unix_time=3000.0))
    evaluation = dict(path=row['evaluation'], sha256=row['evaluation'], start=dict(copy.deepcopy(header),
                      mode='evaluate_sample', seed=declaration['evaluation_seed'], steps=game['evaluation_actions'],
                      unix_time=4000.0, environment_seeds=[declaration['evaluation_seed'] + stream * 1_000_003
                                                        for stream in range(streams)]),
                      accounting=dict(actions=game['evaluation_actions'], budget_complete=True, updates=0))
    return training, evaluation


@pytest.mark.parametrize('streams', [4, 6, 8])
def test_fixed_five_game_declaration_and_all_input_pins(tmp_path, streams):
    declaration = declaration_fixture(tmp_path, streams)
    campaign.verify_declaration(declaration)
    campaign.verify_inputs(declaration)
    assert len(declaration['runs']) == 15
    for row in declaration['runs']:
        training, evaluation = run_fixture(declaration, row)
        campaign.verify_run_declaration(declaration, row, training, evaluation)


@pytest.mark.parametrize('mutation, message', [
    (lambda plan: plan.update(protocol='kindle-boxing-ratio-pilot-v1'), 'unsupported replication'),
    (lambda plan: plan.update(protocol='kindle-atari-five-replication-v1'), 'unsupported replication'),
    (lambda plan: plan.update(training_seeds=[0, 1, 2]), 'fresh replication seeds'),
    (lambda plan: plan.update(training_seeds=[1009, 1009, 3019]), 'fresh replication seeds'),
    (lambda plan: plan.update(evaluation_streams=1), 'stream count'),
    (lambda plan: plan.update(training_streams=True, evaluation_streams=True), 'stream count'),
    (lambda plan: plan.update(training_streams=6, evaluation_streams=6.0), 'stream count'),
    (lambda plan: plan.update(training_streams=5, evaluation_streams=5), 'stream count'),
    (lambda plan: plan.update(runtime_gate={}), 'runtime gate inputs'),
    (lambda plan: plan.update(declared_unix_time=float('nan')), 'declaration time'),
    (lambda plan: plan.update(evaluation_seed=True), 'environment seed'),
    (lambda plan: plan['games'].pop('ALE/Qbert-v5'), 'all five'),
    (lambda plan: plan['games']['ALE/Boxing-v5']['criteria'].update(minimum_mean_return=0), 'acceptance criteria'),
    (lambda plan: plan['games']['ALE/Pong-v5'].update(training_actions=0), 'action budget'),
    (lambda plan: plan['games']['ALE/Pong-v5'].update(evaluation_actions=400001), 'action budget'),
    (lambda plan: plan['games']['ALE/Pong-v5'].update(evaluation_actions=True), 'action budget'),
    (lambda plan: plan['header'].pop('cpu_worker_threads'), 'header identity'),
    (lambda plan: plan['header'].update(sticky_actions=0.25), 'Atari protocol'),
    (lambda plan: plan['header']['model_provenance']['perception'].update(kind='dinov3'), 'frontend'),
    (lambda plan: plan['config'].update(seed=0), 'seed/action/reward'),
    (lambda plan: plan['config'].update(intrinsic_reward_scale=1), 'seed/action/reward'),
    (lambda plan: plan['runs'].pop(), 'exactly one run'),
    (lambda plan: plan['runs'].append(copy.deepcopy(plan['runs'][0])), 'exactly one run'),
    (lambda plan: plan['runs'][1].update(seed=1009), 'exactly one run'),
    (lambda plan: plan['runs'][1].update(training=plan['runs'][0]['training']), 'reused run artifact'),
    (lambda plan: plan['runs'][0].update(evaluation=plan['runs'][0]['training']), 'reused run artifact'),
    (lambda plan: plan['pins'].update({plan['runs'][0]['training']: 'hash'}), 'output is a pinned input'),
])
def test_declaration_rejects_reduced_scope_selection_and_recipe_changes(tmp_path, mutation, message):
    declaration = declaration_fixture(tmp_path)
    mutation(declaration)
    with pytest.raises(ValueError, match=message):
        campaign.verify_declaration(declaration)


@pytest.mark.parametrize('mutation, message', [
    (lambda train, frozen: train['start'].update(seed=0), 'run seed'),
    (lambda train, frozen: frozen['start'].update(num_envs=6), 'run seed/streams'),
    (lambda train, frozen: train['start'].update(environment='ALE/Boxing-v5'), 'game or action mode'),
    (lambda train, frozen: frozen['start'].update(mode='evaluate_greedy'), 'game or action mode'),
    (lambda train, frozen: train['start'].update(steps=100000), 'action budget'),
    (lambda train, frozen: train['accounting'].update(actions=100000), 'action budget'),
    (lambda train, frozen: frozen['accounting'].update(budget_complete=False), 'action budget'),
    (lambda train, frozen: frozen['start'].update(environment_seeds=[0] * 8), 'stream seed rule'),
    (lambda train, frozen: train['start']['config'].update(batch_size=32), 'complete declared training config'),
    (lambda train, frozen: frozen['start']['config'].update(seed=0), 'complete declared training config'),
    (lambda train, frozen: train['start']['config']['loss_scales'].update(reconstruction=1), 'complete declared training config'),
    (lambda train, frozen: train['start'].update(native_extension_sha256='wrong'), 'declared header'),
    (lambda train, frozen: train['start'].update(policy_seed_rule='adjacent'), 'declared header'),
    (lambda train, frozen: train['start'].update(unix_time=1000), 'predates its declaration'),
    (lambda train, frozen: train['start'].update(unix_time=float('nan')), 'predates its declaration'),
    (lambda train, frozen: train['start'].update(restored_checkpoint={}), 'fresh training'),
    (lambda train, frozen: train['start'].update(starting_environment_step=8), 'fresh training'),
    (lambda train, frozen: train['start'].update(starting_learner_step=1), 'fresh training'),
    (lambda train, frozen: train['accounting'].update(updates=0), 'fresh training'),
    (lambda train, frozen: frozen['accounting'].update(updates=1), 'not frozen'),
    (lambda train, frozen: frozen['start'].update(unix_time=2500), 'precedes final training'),
])
def test_actual_runs_must_match_the_full_declaration(tmp_path, mutation, message):
    declaration = declaration_fixture(tmp_path)
    row = declaration['runs'][0]
    training, evaluation = run_fixture(declaration, row)
    mutation(training, evaluation)
    with pytest.raises(ValueError, match=message):
        campaign.verify_run_declaration(declaration, row, training, evaluation)


@pytest.mark.parametrize('role', ['runner', 'wrapper', 'native', 'encoder'])
def test_executable_and_encoder_pins_bind_to_the_header(tmp_path, role):
    declaration = declaration_fixture(tmp_path)
    other = tmp_path / 'different-input'
    other.write_text('different input')
    declaration['pins'][str(other)] = campaign.sha256(other)
    declaration['inputs'][role] = str(other)
    with pytest.raises(ValueError, match='declared executable differs|declared encoder differs'):
        campaign.verify_inputs(declaration)


def test_changed_or_unpinned_auditor_fails(tmp_path):
    declaration = declaration_fixture(tmp_path)
    path = declaration['inputs']['encoder']
    Path(path).write_text('changed')
    with pytest.raises(ValueError, match='changed declared input'):
        campaign.verify_inputs(declaration)
    declaration['pins'][path] = campaign.sha256(path)
    declaration['pins'].pop(str(Path(campaign.matches.__file__).resolve()))
    with pytest.raises(ValueError, match='unpinned auditor'):
        campaign.verify_inputs(declaration)


@pytest.mark.parametrize('mutation, message', [
    (lambda gate: gate.update(protocol='different'), 'runtime gate protocol'),
    (lambda gate: gate['pairs'][0].update(memory_gate_passed=False), 'failed runtime gate'),
    (lambda gate: gate['pairs'][0].update(repeated_state_and_traces_exact=False), 'failed runtime gate'),
    (lambda gate: gate['runs'].pop(), 'reverse-order runtime repeats'),
    (lambda gate: gate['runs'][0].update(train_trace_sha256='changed'), 'state or trace differs'),
    (lambda gate: gate['runs'][0].update(restore_trace_sha256='changed'), 'state or trace differs'),
    (lambda gate: gate['runs'][0]['state'].update(extra='changed'), 'state or trace differs'),
    (lambda gate: gate['runs'][0]['checkpoint'].update(finite_and_complete=False), 'invalid runtime checkpoint'),
    (lambda gate: gate['runs'][0]['training']['accounting'].update(actions=100), 'incomplete runtime work'),
    (lambda gate: gate['runs'][0]['restore']['accounting'].update(updates=1), 'incomplete runtime work'),
    (lambda gate: gate['runs'][0]['train_phase'].update(command_end=1100), 'precede replication declaration'),
    (lambda gate: gate['runs'][0]['restore_phase'].update(command_start=700), 'precede replication declaration'),
    (lambda gate: gate['runs'][0]['train_gpu'].update(minimum_reported_free_mib=1631), 'directly free memory'),
    (lambda gate: gate['runs'][0]['restore_gpu'].update(minimum_reported_free_mib=2047), 'directly free memory'),
    (lambda gate: gate['runs'][0]['train_gpu'].update(minimum_reported_free_mib=2048.0), 'directly free memory'),
    (lambda gate: gate['runs'][0]['restore_gpu'].update(coverage_passed=False), 'sample coverage'),
    (lambda gate: gate['runs'][1]['train_gpu'].update(free_memory_gate_passed=False), 'directly free memory'),
])
def test_runtime_gate_rejects_failed_memory_incomplete_work_and_changed_repeats(tmp_path, mutation, message):
    declaration = declaration_fixture(tmp_path, streams=6)
    path = Path(declaration['runtime_gate']['summary'])
    result = json.loads(path.read_text())
    mutation(result)
    path.write_text(json.dumps(result))
    declaration['pins'][str(path)] = campaign.sha256(path)
    with pytest.raises(ValueError, match=message):
        campaign.verify_runtime_gate(declaration)


def test_runtime_gate_cannot_be_reused_for_a_different_learner(tmp_path):
    declaration = declaration_fixture(tmp_path, streams=4)
    declaration['config']['train_ratio'] = 256
    with pytest.raises(ValueError, match='different learner'):
        campaign.verify_runtime_gate(declaration)


def test_runtime_gate_pins_its_own_inputs_and_completed_result(tmp_path):
    declaration = declaration_fixture(tmp_path)
    path = Path(declaration['runtime_gate']['summary'])
    original = path.read_text()
    path.write_text(original + '\n')
    with pytest.raises(ValueError, match='unpinned runtime gate'):
        campaign.verify_runtime_gate(declaration)
    path.write_text(original)
    Path(declaration['inputs']['native']).write_text('changed native')
    with pytest.raises(ValueError, match='changed runtime gate input'):
        campaign.verify_runtime_gate(declaration)


def campaign_flow_fixture(tmp_path, monkeypatch, *, failing=False, repeated=False):
    """Stub lower-level guards to test orchestration, never as trained-model evidence."""
    declaration = declaration_fixture(tmp_path)
    path = tmp_path / 'declaration.json'
    path.write_text(json.dumps(declaration))
    runs, calls = {}, []
    for row in declaration['runs']:
        training, evaluation = run_fixture(declaration, row)
        evaluation['episodes'] = []
        runs[row['training']], runs[row['evaluation']] = training, evaluation
        Path(row['replay']).write_text(json.dumps(dict(episodes=[])))
    monkeypatch.setattr(campaign, 'read_run', runs.__getitem__)
    monkeypatch.setattr(campaign, 'verify_inputs', lambda plan: calls.append('pins'))
    monkeypatch.setattr(campaign, 'check_match_replay', lambda run, replay: calls.append('match replay'))
    monkeypatch.setattr(campaign.tasks, 'check_replay', lambda run, replay: calls.append('task replay'))
    monkeypatch.setattr(campaign.matches, 'score_matches',
                        lambda game, rows: dict(mastery_passed=True))
    monkeypatch.setattr(campaign.tasks, 'score_tasks',
                        lambda game, rows: dict(task_gate_passed=not (failing and game == 'ALE/Qbert-v5')))

    def checkpoint_guard(checkpoint, training, evaluation, schema):
        calls.append(('checkpoint', checkpoint, schema))
        return dict(tensor_sha256={name: name + ('duplicate' if repeated else checkpoint)
                                  for name in ('world', 'behavior', 'slow_value')})

    monkeypatch.setattr(campaign.matches, 'verify_checkpoint', checkpoint_guard)
    return path, calls


@pytest.mark.parametrize('failing', [False, True])
def test_all_fifteen_individual_gates_are_required(tmp_path, monkeypatch, failing):
    path, calls = campaign_flow_fixture(tmp_path, monkeypatch, failing=failing)
    result = campaign.audit_campaign(path)
    assert result['replication_passed'] is not failing
    assert 'five_game_goal_complete' not in result
    assert result['campaign_declaration_verified'] and result['reliability_assessed']
    assert len(result['results']) == 15
    assert calls.count('pins') == 2
    assert calls.count('match replay') == 6 and calls.count('task replay') == 9
    assert len([call for call in calls if isinstance(call, tuple)]) == 15
    assert result['declaration']['sha256'] == campaign.sha256(path)


def test_relabeling_one_checkpoint_as_multiple_models_fails(tmp_path, monkeypatch):
    path, _ = campaign_flow_fixture(tmp_path, monkeypatch, repeated=True)
    with pytest.raises(ValueError, match='reused trained tensor state'):
        campaign.audit_campaign(path)


def test_two_successful_seeds_cannot_hide_the_third_failure(tmp_path, monkeypatch):
    path, _ = campaign_flow_fixture(tmp_path, monkeypatch)
    scores = iter([True, False, True, True, True, True])
    monkeypatch.setattr(campaign.matches, 'score_matches',
                        lambda game, rows: dict(mastery_passed=next(scores)))
    result = campaign.audit_campaign(path)
    assert sum(row['passed'] for row in result['results']) == 14
    assert not result['replication_passed']


def test_checkpoint_guard_failure_cannot_become_campaign_success(tmp_path, monkeypatch):
    path, _ = campaign_flow_fixture(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError('missing final checkpoint tensor')

    monkeypatch.setattr(campaign.matches, 'verify_checkpoint', reject)
    with pytest.raises(ValueError, match='missing final checkpoint tensor'):
        campaign.audit_campaign(path)


def match_replay_fixture(tmp_path):
    declaration = declaration_fixture(tmp_path)
    row = next(row for row in declaration['runs'] if row['environment'] == 'ALE/Boxing-v5')
    _, evaluation = run_fixture(declaration, row)
    header = evaluation['start']
    path = Path(evaluation['path'])
    path.write_text('Synthetic replay-binding unit fixture, not an executable game log.\n')
    evaluation['sha256'] = campaign.sha256(path)
    episodes = [dict(stream=stream, episode=0, episode_return=60, terminated=True, truncated=False)
                for stream in range(8)]
    replay_episodes = []
    for episode in episodes:
        task = dict(task='natural_match_win', episode_score=60, episode_frames=100,
                    terminated=True, truncated=False, eligible_completed_episode=True, episode_success=True)
        replay_episodes.append(dict(episode, task_outcome=task, first_frame=0, last_frame=100))
    evaluation.update(episodes=episodes, end=dict(partial_returns=[0] * 8, executed_action_frames=[104] * 8))
    manifest = tmp_path / 'replay-manifest.json'
    manifest.write_text(json.dumps(dict(environment=row['environment'], pins=declaration['pins'])))
    replay = dict(protocol='kindle-atari-task-replay-v1', source_log=str(path),
                  source_log_sha256=evaluation['sha256'], source_header=copy.deepcopy(header),
                  source_accounting=copy.deepcopy(evaluation['accounting']), agent_constructed=False,
                  learner_updates=0, full_trajectory_replayed=True, source_manifest=str(manifest),
                  source_manifest_sha256=campaign.sha256(manifest),
                  rom=campaign.check_atari_adapter.rom_identity(row['environment']),
                  ale_native_sha256=campaign.sha256(campaign.ale_native.__file__),
                  wrapper_sha256=header['wrapper_sha256'], replay_script_sha256=campaign.sha256(campaign.replay_atari.__file__),
                  observer_sha256=campaign.sha256(campaign.atari_tasks.__file__), episodes=replay_episodes,
                  partial=[dict(stream=stream, terminated=False, truncated=False, eligible_completed_episode=False,
                                episode_success=False, episode_score=0, episode_frames=4) for stream in range(8)],
                  task_successes=8)
    return evaluation, replay


def test_match_replay_binding_and_partial_tail_checks(tmp_path):
    evaluation, replay = match_replay_fixture(tmp_path)
    campaign.check_match_replay(evaluation, replay)


@pytest.mark.parametrize('mutation, message', [
    (lambda replay: replay.update(source_log_sha256='wrong'), 'source differs'),
    (lambda replay: replay['source_header'].update(seed=0), 'source differs'),
    (lambda replay: replay['source_accounting'].update(actions=8), 'source differs'),
    (lambda replay: replay.update(agent_constructed=True), 'complete CPU'),
    (lambda replay: replay.update(learner_updates=1), 'complete CPU'),
    (lambda replay: replay.update(full_trajectory_replayed=False), 'complete CPU'),
    (lambda replay: replay.update(source_manifest_sha256='wrong'), 'manifest changed'),
    (lambda replay: replay['rom'].update(sha256='wrong'), 'ROM differs'),
    (lambda replay: replay.update(observer_sha256='wrong'), 'implementation changed'),
    (lambda replay: replay['episodes'].pop(), 'episode count differs'),
    (lambda replay: replay['episodes'][0].update(episode_return=100), 'episodes differ'),
    (lambda replay: replay['episodes'][0].update(first_frame=1), 'frames differ'),
    (lambda replay: replay['episodes'][0]['task_outcome'].update(episode_frames=101), 'frames differ'),
    (lambda replay: replay['episodes'][0]['task_outcome'].update(episode_score=61), 'task differs'),
    (lambda replay: replay['episodes'][0]['task_outcome'].update(eligible_completed_episode=False), 'task differs'),
    (lambda replay: replay['episodes'][0]['task_outcome'].update(episode_success=False), 'task differs'),
    (lambda replay: replay['partial'].pop(), 'missing match tails'),
    (lambda replay: replay['partial'].reverse(), 'missing match tails'),
    (lambda replay: replay['partial'][0].update(episode_frames=3), 'tail differs'),
    (lambda replay: replay['partial'][0].update(episode_score=1), 'tail differs'),
    (lambda replay: replay['partial'][0].update(eligible_completed_episode=True), 'tail differs'),
    (lambda replay: replay.update(task_successes=9), 'success count differs'),
])
def test_match_replay_divergence_fails(tmp_path, mutation, message):
    evaluation, replay = match_replay_fixture(tmp_path)
    mutation(replay)
    with pytest.raises(ValueError, match=message):
        campaign.check_match_replay(evaluation, replay)

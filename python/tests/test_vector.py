import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import kindle
from kindle._vector_audit import audit, require_numbers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector
import audit_pong
import profile_atari_vector


def test_vector_api_rejects_invalid_config_before_loading_weights():
    with pytest.raises(ValueError, match="positive"):
        kindle.VectorAgent("unused", 0, {})
    config = kindle.default_config(18)
    config["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        kindle.VectorAgent("unused", 4, config)


@pytest.mark.parametrize("encoder", ["dinov3", "levjepa"])
def test_vector_encoder_selection_validates_config_before_gpu(encoder):
    config = kindle.default_config(18)
    config["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        kindle.VectorAgent("unused", 2, config, encoder=encoder)


def test_vector_api_rejects_unknown_encoder_before_loading_weights():
    with pytest.raises(ValueError, match="encoder must be dinov3 or levjepa"):
        kindle.VectorAgent("unused", 2, {}, encoder="unknown")


@pytest.mark.parametrize("args, message", [
    (["--steps", "7", "--num-envs", "2"], "multiple"),
    (["--num-envs", "0"], "multiple"),
    (["--greedy"], "frozen evaluation"),
    (["--world-microbatch-size", "0"], "must be positive"),
    (["--restore", "unused", "--batch-size", "32"], "overrides require a fresh run"),
    (["--restore", "unused", "--encoder", "dinov3"], "restore selects the checkpoint encoder"),
])
def test_runner_rejects_ambiguous_budgets_before_gpu(monkeypatch, capsys, tmp_path, args, message):
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "unused", "--output", str(tmp_path / "log.jsonl"), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("encoder,restore", [(None, False), ("dinov3", False), ("levjepa", False), (None, True)])
def test_runner_passes_explicit_frontend_or_restores_identity(monkeypatch, tmp_path, encoder, restore):
    calls, environments = [], []

    class BeforeGpu(Exception):
        pass

    class Agent:
        def __new__(cls, weights, streams, config, *, encoder):
            calls.append(("new", weights, streams, encoder))
            raise BeforeGpu

        @classmethod
        def restore(cls, checkpoint, weights, streams):
            calls.append(("restore", checkpoint, weights, streams))
            raise BeforeGpu

    def make(*_, **__):
        env = SimpleNamespace(action_space=SimpleNamespace(n=18), closed=False)
        env.reset = lambda **_: (None, {})
        env.close = lambda: setattr(env, "closed", True)
        environments.append(env)
        return env

    output = tmp_path / "log.jsonl"
    args = ["atari_vector.py", "weights", "--output", str(output), "--num-envs", "2"]
    if encoder is not None:
        args += ["--encoder", encoder]
    if restore:
        args += ["--restore", "saved"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(atari_vector.gym, "make", make)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    monkeypatch.setattr(atari_vector, "checkpoint_identity", lambda _: {"fixture": True})
    monkeypatch.setattr(kindle, "VectorAgent", Agent)
    with pytest.raises(BeforeGpu):
        atari_vector.main()
    expected = ("restore", "saved", "weights", 2) if restore else ("new", "weights", 2, encoder or "levjepa")
    assert calls == [expected]
    assert len(environments) == 2 and all(env.closed for env in environments)
    assert output.read_text() == ""  # no run_start claims for a failed construction


@pytest.mark.parametrize("args, message", [
    (["--batch-size", "0"], "must be positive"),
    (["--num-envs", "2", "2"], "distinct"),
    (["--batch-size", "64"], "warmup"),
])
def test_profiler_rejects_unusable_windows_before_starting_jobs(monkeypatch, capsys, tmp_path, args, message):
    directory = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["profile_atari_vector.py", "unused", str(directory), *args])
    with pytest.raises(SystemExit) as error:
        profile_atari_vector.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not directory.exists()


@pytest.mark.parametrize("encoder", ["levjepa", "dinov3"])
def test_profiler_retains_failed_jobs_without_claiming_a_completed_matrix(monkeypatch, tmp_path, encoder):
    directory = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["profile_atari_vector.py", "unused", str(directory), "--num-envs", "2", "--encoder", encoder])
    monitor = SimpleNamespace(terminate=lambda: None, wait=lambda **_: None)
    monkeypatch.setattr(profile_atari_vector.subprocess, "Popen", lambda *_, **__: monitor)
    monkeypatch.setattr(profile_atari_vector.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=1))
    with pytest.raises(SystemExit) as error:
        profile_atari_vector.main()
    assert error.value.code == 1
    results = json.loads((directory / "summary.json").read_text())
    assert results[0]["status"] == "failed"
    assert results[0]["num_envs"] == 2 and results[0]["exit_code"] == 1
    assert "actions_per_second" not in results[0]
    command = results[0]["command"]
    assert command[command.index("--encoder") + 1] == encoder


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.0", True])
def test_learner_metrics_cannot_hide_nonfinite_values_as_null(bad):
    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        require_numbers({"world": {"total_loss": bad}})


def fixture_events():
    config = dict(batch_size=1, batch_length=2, replay_context=1, replay_capacity=32, train_ratio=2.0, action_count=3)
    events = [dict(event="run_start", protocol="kindle-vector-v1", num_envs=2, steps=6,
                   environment_seeds=[0, 1000003], mode="train", config=config,
                   starting_learner_step=0, starting_environment_step=0, action_repeat=4)]
    updates = 0
    for tick in range(1, 4):
        events.append(dict(event="transition", run_step=tick * 2, vector_tick=tick,
                           actions=[1, 2], rewards=[1.0, 0.0], stored_rewards=[[1.0, 0.0], [0.0, 0.0]],
                           terminated=[tick == 2, False], truncated=[False, False],
                           executed_action_frames=[tick * 4, tick * 4]))
        for _ in range(0 if tick == 1 else 1 if tick == 2 else 2):
            updates += 1
            events.append(dict(event="learner", run_step=tick * 2,
                               report=dict(learner_step=updates, replay_len=2 + tick * 2 + int(tick == 3),
                                           world={"total_loss": 1.0}, behavior={"total_loss": 2.0}, timing={"total_seconds": 0.5})))
        if tick == 2:
            events.append(dict(event="episode", stream=0, run_step=4, stream_step=2, episode=0,
                               episode_return=2.0, episode_length=2, terminated=True, truncated=False))
            events.append(dict(event="reset", run_step=4, streams=[0]))
    events.append(dict(event="run_end", run_step=6, vector_ticks=3, environment_step=6, learner_step=3,
                       replay_len=9, training_debt=0.0, executed_action_frames=[12, 12],
                       total_rewards=[3.0, 0.0], episode_counts=[1, 0], partial_returns=[1.0, 0.0],
                       partial_lengths=[1, 3], stage_seconds={"act": 0.1}, elapsed_seconds=2.0,
                       actions_per_second=3.0, aggregate_simulated_wall_ratio=0.2,
                       per_stream_simulated_wall_ratio=[0.1, 0.1],
                       reason="budget_complete", learner_updates=3, completed_games=1,
                       natural_wins=1, mean_completed_return=2.0))
    return events


def write_log(tmp_path, events):
    path = tmp_path / "vector.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def test_vector_accounting_keeps_reset_records_out_of_action_credit(tmp_path):
    result = audit(write_log(tmp_path, fixture_events()))
    assert result["actions"] == 6
    assert result["updates"] == 3
    assert result["natural_wins"] == 1


@pytest.mark.parametrize("mutation", ["lost_update", "wrong_reset", "null_loss", "wrong_reward", "ticks_as_actions", "aggregate_as_per_stream", "missing_end"])
def test_vector_audit_rejects_broken_ledgers(tmp_path, mutation):
    events = copy.deepcopy(fixture_events())
    if mutation == "lost_update":
        events.pop(next(i for i, e in enumerate(events) if e["event"] == "learner"))
    elif mutation == "wrong_reset":
        next(e for e in events if e["event"] == "reset")["streams"] = [1]
    elif mutation == "null_loss":
        next(e for e in events if e["event"] == "learner")["report"]["world"]["total_loss"] = None
    elif mutation == "wrong_reward":
        events[-1]["total_rewards"][0] = 4.0
    elif mutation == "ticks_as_actions":
        events[1]["run_step"] = 1
    elif mutation == "aggregate_as_per_stream":
        events[-1]["per_stream_simulated_wall_ratio"] = [0.2, 0.2]
    else:
        events.pop()
    with pytest.raises(ValueError):
        audit(write_log(tmp_path, events))


def mastery_events():
    events = fixture_events()
    events[0].update(model_provenance={"perception": {}}, trainable_parameter_counts={"world": 1})
    events[-1].update(reset_noop_frames=[0, 0], emulator_resets=[2, 1])
    return events


def test_vector_mastery_reader_keeps_unfinished_tails_and_per_stream_clocks(tmp_path):
    result = audit_pong.audit_vector_run(write_log(tmp_path, mastery_events()))
    assert result["natural_games"] == result["natural_wins"] == 1
    assert result["timeouts"] == 0
    assert result["mean_return"] == 2
    assert result["end"]["partial_returns"] == [1, 0]
    assert result["simulated_to_wall"] == 0.2
    assert result["per_stream_simulated_to_wall"] == [0.1, 0.1]


def test_vector_mastery_timeout_is_not_a_win_or_dropped_from_score(tmp_path):
    events = mastery_events()
    transition = next(e for e in events if e["event"] == "transition" and e["run_step"] == 4)
    transition["terminated"][0] = False
    transition["truncated"][0] = True
    next(e for e in events if e["event"] == "episode").update(terminated=False, truncated=True)
    events[-1]["natural_wins"] = 0
    result = audit_pong.audit_vector_run(write_log(tmp_path, events))
    assert result["natural_games"] == result["natural_wins"] == result["win_fraction"] == 0
    assert result["timeouts"] == 1 and result["mean_return"] == 2


def test_vector_mastery_rejects_shaping_even_when_reward_ledgers_balance(tmp_path):
    events = mastery_events()
    events[1]["rewards"][0] = events[1]["stored_rewards"][0][0] = 2.0
    next(e for e in events if e["event"] == "episode")["episode_return"] = 3.0
    events[-1].update(total_rewards=[4.0, 0.0], mean_completed_return=3.0)
    assert audit(write_log(tmp_path, events))["accounting_valid"]
    with pytest.raises(ValueError, match="unshaped Pong"):
        audit_pong.audit_vector_run(write_log(tmp_path, events))


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows[-1].update(reason="interrupted"), "interrupted"),
    (lambda rows: rows[-1].update(emulator_resets=[3, 1]), "clock/reset"),
    (lambda rows: rows[1]["stored_rewards"][0].__setitem__(1, 0.5), "intrinsic reward"),
])
def test_vector_mastery_rejects_interruption_extra_resets_and_intrinsic_reward(tmp_path, mutation, message):
    events = mastery_events()
    mutation(events)
    with pytest.raises(ValueError, match=message):
        audit_pong.audit_vector_run(write_log(tmp_path, events))


def vector_protocol_run():
    perception = dict(kind="levjepa", checkpoint_sha256=audit_pong.ENCODER_SHA256,
                      model_id="galilai-group/LeVJEPA-VideoMix-Large",
                      checkpoint_revision="e831a0347737fcaa660b39c57d41c109de399845",
                      encoding_revision="levjepa-large-f32-chunk16-letterbox224-jl64-pool2-v1")
    return dict(path="fixture", start=dict(protocol="kindle-vector-v1", environment="ALE/Pong-v5",
        ale_py_version="0.12.1", atari_protocol="published", action_repeat=4,
        full_action_space=True, noop_max=0, max_episode_frames=100_000,
        mode="train", steps=200_000, seed=0, num_envs=4, sticky_actions=0.0,
        environment_seeds=[0, 1000003, 2000006, 3000009],
        policy_seed_rule="config.seed + stream (wrapping u64)",
        config={"seed": 0, "action_count": 18}, action_meanings=list(range(18)),
        perception=perception, model_provenance={"perception": perception}))


def test_vector_mastery_protocol_preserves_declared_aggregate_budget_and_frontend():
    audit_pong.validate_protocol(vector_protocol_run(), "train", 200_000, 4)


def test_dino_control_cannot_be_relabelled_as_levjepa_mastery():
    run = vector_protocol_run()
    run["start"]["perception"]["kind"] = "dinov3"
    with pytest.raises(ValueError, match="not the pinned LeVJEPA experiment"):
        audit_pong.validate_protocol(run, "train", 200_000, 4)


@pytest.mark.parametrize("field, value", [
    ("steps", 50_000), ("num_envs", 2), ("mode", "evaluate_greedy"),
    ("environment_seeds", [0, 0, 0, 0]), ("sticky_actions", 0.25),
    ("policy_seed_rule", "shared"), ("noop_max", 30),
])
def test_vector_mastery_protocol_rejects_changed_collection(field, value):
    run = vector_protocol_run()
    run["start"][field] = value
    with pytest.raises(ValueError):
        audit_pong.validate_protocol(run, "train", 200_000, 4)

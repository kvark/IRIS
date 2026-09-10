"""Synthetic multi-match diagnostics, never policy or GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from kindle._vector_audit import episode_summary

from test_episode_evaluation import frozen_run
from test_world_probe import probe, recorded_rows, write_rows


@pytest.mark.parametrize("target", [None, 2])
def test_selects_first_matches_without_relabelling_vector_source(frozen_run, target):
    path, rows = frozen_run(target=target, cap=14 if target is None else 100)
    header, transitions, record = probe.recorded_matches(path, 2)
    assert header == rows[0] and header["num_envs"] == 2
    assert record["source_num_envs"] == 2 and record["stream"] == 0
    assert record["source_accounting"]["budget_complete"]
    assert record["source_accounting"]["updates"] == 0
    assert record["selected_actions"] == len(transitions) == 4
    assert [row["run_step"] for row in transitions] == [2, 4, 6, 8]
    assert [row["episode"] for row in record["episodes"]] == [0, 1]
    assert [row["episode_return"] for row in record["episodes"]] == [-2.0, -2.0]
    assert all(len(row["actions"]) == 1 for row in transitions)
    raw = path.read_bytes()
    prefix = raw[:record["prefix_bytes"]]
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["prefix_sha256"] == hashlib.sha256(prefix).hexdigest()
    assert json.loads(prefix.splitlines()[-1]) == record["episodes"][-1]
    assert len(prefix) < len(raw)


@pytest.mark.parametrize("mutation", ["bad_tail", "learning", "greedy", "missing_end", "missing_reset", "seed"])
def test_full_recording_must_verify_even_after_selected_prefix(frozen_run, mutation):
    path, rows = frozen_run()
    if mutation == "bad_tail":
        next(row for row in reversed(rows) if row["event"] == "transition")["actions"][1] = 99
    elif mutation == "learning":
        rows.insert(-1, dict(event="learner", run_step=rows[-1]["run_step"]))
    elif mutation == "greedy":
        rows[0]["mode"] = "evaluate_greedy"
    elif mutation == "missing_end":
        rows.pop()
    elif mutation == "missing_reset":
        rows.pop(next(i for i, row in enumerate(rows) if row["event"] == "reset"))
    else:
        rows[0]["environment_seeds"][0] += 1
    write_rows(path, rows)
    with pytest.raises(ValueError):
        probe.recorded_matches(path, 1)


@pytest.mark.parametrize("count", [0, -1, True, 0.5, 4])
def test_invalid_or_unavailable_match_count_is_not_shortened(frozen_run, count):
    path, _ = frozen_run()
    with pytest.raises(ValueError):
        probe.recorded_matches(path, count)


def test_incomplete_episode_budget_is_not_a_valid_recording(frozen_run):
    path, _ = frozen_run(cap=10)
    with pytest.raises(ValueError, match="incomplete declared run budget"):
        probe.recorded_matches(path, 1)


def test_selection_retains_cutoffs_and_negative_matches(frozen_run):
    path, rows = frozen_run()
    completed = []
    for row in rows:
        if row["event"] == "transition" and row["run_step"] == 4:
            row["terminated"][0], row["truncated"][0] = False, True
        if row["event"] == "episode":
            if row["stream"] == row["episode"] == 0:
                row.update(terminated=False, truncated=True)
            completed.append(row)
        if row["event"] in ("progress", "run_end"):
            row.update(episode_summary(completed))
    write_rows(path, rows)
    _, _, record = probe.recorded_matches(path, 2)
    assert record["episodes"][0]["truncated"]
    assert record["episodes"][0]["episode_return"] == -2


def test_source_content_change_during_selection_is_refused(frozen_run, monkeypatch):
    path, _ = frozen_run()
    monkeypatch.setattr(probe, "sha256_file", lambda path: "changed")
    with pytest.raises(ValueError, match="recording changed"):
        probe.recorded_matches(path, 2)


@pytest.fixture
def multi_replay(frozen_run, tmp_path, monkeypatch):
    path, rows = frozen_run()
    header = rows[0]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hashes = {}
    for name in ("metadata.json", "world.safetensors", "behavior.safetensors", "slow_value.safetensors"):
        hashes[name] = hashlib.sha256(b"synthetic").hexdigest()
        (checkpoint / name).write_bytes(b"synthetic")
    header["restored_checkpoint"] = dict(metadata_sha256=hashes["metadata.json"],
        tensor_sha256={name: hashes[f"{name}.safetensors"] for name in ("world", "behavior", "slow_value")})
    write_rows(path, rows)
    transitions = [row for row in rows if row["event"] == "transition"]
    events = []

    class Environment:
        action_space = SimpleNamespace(n=2)
        action_meanings = header["action_meanings"]
        executed_action_frames = 0
        index = 0
        episode = -1
        closed = False

        def reset(self, *, seed=None):
            assert seed == (header["seed"] if self.index == 0 else None)
            self.episode += 1
            self.offset = 0
            events.append(("reset", self.index))
            return self.episode * 10, {}

        def step(self, action):
            row = transitions[self.index]
            assert action == row["actions"][0]
            events.append(("environment", self.index))
            self.index += 1
            self.offset += 1
            self.executed_action_frames = row["executed_action_frames"][0]
            return (self.episode * 10 + self.offset, row["rewards"][0],
                    row["terminated"][0], row["truncated"][0], {})

        def close(self):
            self.closed = True

    class Agent:
        config = copy.deepcopy(header["config"])
        provenance = header["model_provenance"]
        gpu_device = {"synthetic": True}
        environment_step = header["starting_environment_step"]
        learner_step = header["starting_learner_step"]
        frame = 0
        wrong_action = False
        horizons = []

        def begin_episode(self, frame):
            self.frame = frame
            events.append(("begin", environment.index))

        @property
        def visual_observation(self):
            return [float(self.frame)]

        def act(self, action_mask=None):
            return action_mask.index(True) if action_mask else int(self.wrong_action)

        def posterior_action_probabilities(self):
            return [0.4, 0.6]

        def posterior_value_prediction(self):
            return 2.0

        def observation_prediction(self):
            return self.visual_observation

        def prior_diagnostic_rollout(self, actions):
            events.append(("forecast", environment.index))
            self.horizons.append(len(actions))
            return [0.5] * len(actions), [[float(self.frame + i + 1)] for i in range(len(actions))]

        def prior_behavior_rollout(self, actions):
            events.append(("forecast", environment.index))
            return [0.5] * len(actions), [0.9] * len(actions), [0.0] * len(actions)

        def observe(self, frame, **transition):
            self.frame = frame
            self.environment_step += 1

        def posterior_reward_prediction(self):
            return -1.0

    environment, agent = Environment(), Agent()
    monkeypatch.setattr(probe.gym, "make", lambda *a, **kw: environment)
    monkeypatch.setattr(probe, "DreamerAtariPreprocessing", lambda env, **kw: env)
    monkeypatch.setattr(probe.kindle, "Agent", SimpleNamespace(restore=lambda *a: agent))
    encoder = tmp_path / "encoder"
    encoder.write_bytes(b"synthetic")
    output, trace = tmp_path / "result.json", tmp_path / "trace.jsonl"
    monkeypatch.setattr(sys, "argv", ["probe", str(encoder), str(checkpoint), "--recorded-run", str(path),
        "--recorded-episodes", "2", "--horizon", "3", "--stride", "1", "--output", str(output), "--trace", str(trace)])
    return environment, agent, events, output, trace


@pytest.mark.parametrize("conditioned", [False, True])
def test_multiple_matches_preserve_causality_and_reset_identity(multi_replay, monkeypatch, conditioned):
    environment, agent, events, output, trace = multi_replay
    if conditioned:
        agent.config["seed"] += 1009
        agent.learner_step += 1
        agent.wrong_action = True
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--condition-on-recorded-actions"])
    probe.main()
    result = json.loads(output.read_text())
    assert result["protocol"] == "kindle-world-probe-v4"
    assert result["recorded_game"] is None
    assert result["recorded_matches"]["source_num_envs"] == 2
    assert result["steps"] == 4 and result["completed_episodes"] == 2
    assert result["learner_updates"] == 0 and result["horizon_stops_at_episode_boundary"]
    assert result["sample_count_by_horizon"] == [4, 2, 0]
    assert result["terminal_count_by_horizon"] == [2, 2, 0]
    assert result["prior_observation_mse_by_horizon"] == [0.0, 0.0, None]
    assert result["sampled_actions_match_source"] == (None if conditioned else True)
    assert result["actions_forced_to_source"] == conditioned
    assert [row["after_action"] for row in result["episode_inputs"]] == [0, 2]
    assert result["episode_inputs"][0]["rgb_sha256"] != result["episode_inputs"][1]["rgb_sha256"]
    assert agent.horizons == [2, 2, 1, 1, 2, 2, 1, 1]
    assert events == [("reset", 0), ("begin", 0), *[("forecast", 0)] * 3, ("environment", 0),
        *[("forecast", 1)] * 3, ("environment", 1), ("reset", 2), ("begin", 2),
        *[("forecast", 2)] * 3, ("environment", 2), *[("forecast", 3)] * 3, ("environment", 3)]
    samples = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["episode"] for row in samples] == [0, 0, 0, 1, 1, 1]
    assert all(row["source_run_step"] == row["target_action"] * 2 for row in samples)
    assert all("target_feature_sha256" in row and "target_rgb_sha256" in row for row in samples)
    assert environment.closed


def test_multi_match_strict_mode_still_rejects_action_divergence(multi_replay):
    environment, agent, events, output, trace = multi_replay
    agent.wrong_action = True
    with pytest.raises(ValueError, match="sampled policy diverged"):
        probe.main()
    assert environment.closed and not output.exists()


@pytest.mark.parametrize("conditioned", [False, True])
def test_equal_action_budget_remains_required_with_variable_warmup(conditioned):
    header = recorded_rows()[0]
    agent = SimpleNamespace(config=header["config"], provenance=header["model_provenance"],
        environment_step=11, learner_step=21)
    with pytest.raises(ValueError, match="restored counters differ"):
        probe.verify_recorded_model(header, agent, "expected", {}, conditioned, vector_recording=True)


def test_own_policy_restore_keeps_exact_update_counter():
    header = recorded_rows()[0]
    agent = SimpleNamespace(config=header["config"], provenance=header["model_provenance"],
        environment_step=10, learner_step=21)
    with pytest.raises(ValueError, match="restored counters differ"):
        probe.verify_recorded_model(header, agent, "expected", {}, False, vector_recording=True)


@pytest.mark.parametrize("count", ["0", "-1"])
def test_invalid_cli_count_stops_before_restore(multi_replay, monkeypatch, count):
    args = list(sys.argv)
    args[args.index("--recorded-episodes") + 1] = count
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2 and not multi_replay[2]

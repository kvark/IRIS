import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import audit_pong  # noqa: E402


def events():
    start = {"event": "run_start", "environment": "ALE/Pong-v5", "seed": 0,
             "mode": "evaluate_sample", "steps": 12, "starting_environment_step": 200_000,
             "starting_learner_step": 49_729}
    episodes = [
        {"event": "episode", "episode": 1, "run_step": 4, "length": 4,
         "return": 2, "terminated": True, "truncated": False},
        {"event": "episode", "episode": 2, "run_step": 8, "length": 4,
         "return": 1, "terminated": False, "truncated": True},
    ]
    interval = {"event": "interval", "run_step": 12, "interval_steps": 12,
                "learner_updates": 0, "action_counts": [12] + [0] * 17,
                "reward": 2, "intrinsic_reward": 0, "positive_reward_events": 5,
                "negative_reward_events": 3, "completed_episodes": 2}
    end = {**start, "event": "run_end", "environment_step_delta": 12,
           "environment_step": 200_012, "learner_updates": 0, "learner_step_delta": 0,
           "action_counts": interval["action_counts"], "total_reward": 2,
           "total_intrinsic_reward": 0, "positive_reward_events": 5, "negative_reward_events": 3,
           "completed_episodes": 2, "mean_completed_return": 1.5,
           "partial_episode_return": -1, "partial_episode_length": 4,
           "reset_noop_frames": 0, "emulator_resets": 3, "executed_action_frames": 44,
           "elapsed_seconds": 2.0}
    return [start, *episodes, interval, end]


def write_run(path, data):
    path.write_text("".join(json.dumps(event) + "\n" for event in data))
    return path


def test_timeout_is_not_a_win_or_dropped_from_score(tmp_path):
    result = audit_pong.audit_run(write_run(tmp_path / "run.jsonl", events()))
    assert result["natural_games"] == result["natural_wins"] == result["timeouts"] == 1
    assert result["mean_return"] == 1.5
    assert result["win_fraction"] == 0.5
    assert result["simulated_to_wall"] == pytest.approx(44 / 60 / 2)


def test_pong_serial_scorer_rejects_other_games(tmp_path):
    rows = events()
    rows[0]["environment"] = "ALE/Freeway-v5"
    with pytest.raises(ValueError, match="Pong-only scorer"):
        audit_pong.audit_run(write_run(tmp_path / "run.jsonl", rows))


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "incomplete run"),
    (lambda rows: rows.append(rows[-1]), "after run_end"),
    (lambda rows: rows.insert(1, rows[0]), "multiple run segments"),
    (lambda rows: rows[-1].update(learner_updates=1), "update ledger"),
    (lambda rows: rows[-1].update(total_reward=3), "total_reward ledger"),
    (lambda rows: rows[-1].update(partial_episode_length=3), "episode ledger"),
    (lambda rows: rows[1].update(length=3), "episode counter/length"),
    (lambda rows: rows[2].update(terminated=False, truncated=False), "incomplete episode"),
    (lambda rows: rows[-1].update(elapsed_seconds=float("nan")), "non-finite"),
    (lambda rows: rows[-2].update(interval_steps=11), "interval gap"),
])
def test_audit_rejects_broken_ledgers(tmp_path, mutation, message):
    data = events()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        audit_pong.audit_run(write_run(tmp_path / "run.jsonl", data))


@pytest.mark.parametrize("value", [None, True, "NaN", float("nan"), float("inf")])
def test_learner_metrics_cannot_hide_nonfinite_values_as_json_null(tmp_path, value):
    data = events()
    data.insert(1, {"event": "learner", "run_step": 1, "report": {
        "learner_step": 49_730, "world": {"total_loss": value}}})
    with pytest.raises(ValueError, match="non-numeric|non-finite"):
        audit_pong.audit_run(write_run(tmp_path / "run.jsonl", data))


def test_checkpoint_evaluation_is_bound_to_exact_files(tmp_path):
    metadata = {"format": 3, "architecture": "dreamerv3-visual-features",
                "config": {}, "perception": {}, "environment_step": 200_000,
                "learner_step": 49_729, "tensor_sha256": {}}
    provenance = {key: "revision" for key in ("dreamerv3_revision", "meganeura_revision",
                                             "blade_revision", "future_head_revision")}
    metadata.update(provenance)
    for name in ("world", "behavior", "slow_value"):
        path = tmp_path / f"{name}.safetensors"
        save_file({"parameter": np.ones((2,), dtype=np.float32)}, path)
        metadata["tensor_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    train = {"start": {"config": {}, "perception": {}, "model_provenance": provenance},
             "end": {"learner_updates": 49_729}}
    evaluation = {"start": {"config": {}, "perception": {}, "starting_environment_step": 200_000,
                            "starting_learner_step": 49_729, "restored_checkpoint": {
                                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                                "tensor_sha256": copy.deepcopy(metadata["tensor_sha256"])}}}
    train["checkpoint"] = {"run_step": 200_000, "learner_step": 49_729,
                           "identity": copy.deepcopy(evaluation["start"]["restored_checkpoint"])}
    assert len(audit_pong.audit_checkpoint(tmp_path, train, evaluation)["files"]) == 3
    evaluation["start"]["restored_checkpoint"]["tensor_sha256"]["world"] = "wrong"
    with pytest.raises(ValueError, match="different checkpoint files"):
        audit_pong.audit_checkpoint(tmp_path, train, evaluation)
    train["start"]["num_envs"] = 4
    metadata["collection_streams"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="collection stream count"):
        audit_pong.audit_checkpoint(tmp_path, train, evaluation)

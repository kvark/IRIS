import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import audit_canary


def reports(profile=False):
    row = dict(learner_step=1, replay_len=65,
               **{section: dict.fromkeys(fields, 0.0) for section, fields in audit_canary.METRICS.items()},
               timing=dict.fromkeys(audit_canary.STAGES, 0.1))
    row["world"].update(rewarded_count=2, positive_reward_count=1,
                        negative_reward_count=1, zero_reward_count=1022)
    row["timing"]["total_seconds"] = 1.0
    if profile:
        for name, (calls, size) in audit_canary.TRANSFERS.items():
            phases = audit_canary.READBACK_PHASES if name.endswith("readback") else ["seconds"]
            row["timing"][name] = dict(calls=calls, bytes=size, **dict.fromkeys(phases, 0.001))
        for names in audit_canary.HOST_PHASES.values():
            row["timing"].update(dict.fromkeys(names, 0.001))
    return [{**copy.deepcopy(row), "learner_step": step} for step in range(1, 9)]


def write_reports(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("profile", [False, True])
def test_complete_canary_reports(tmp_path, profile):
    path = write_reports(tmp_path / "reports.jsonl", reports(profile))
    result = audit_canary.audit(path, 8, readback_profile=profile)
    assert result["reports_valid"] and result["reports"] == 8
    assert result["readback_report_fields_validated"] == profile
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "not tensor parity" in result["scope"]


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.clear(), "found 0"),
    (lambda rows: rows.pop(), "found 7"),
    (lambda rows: rows.append(rows[-1]), "found 9"),
    (lambda rows: rows[3].update(learner_step=3), "contiguous"),
    (lambda rows: rows[0].update(learner_step=True), "contiguous"),
    (lambda rows: rows[0].update(learner_step=1.0), "contiguous"),
    (lambda rows: rows[0].update(replay_len=64), "65-record"),
    (lambda rows: rows[0].update(replay_len=65.0), "65-record"),
    (lambda rows: rows[0].pop("world"), "LearnReport"),
    (lambda rows: rows[0].update(event="learner"), "LearnReport"),
    (lambda rows: rows[0]["world"].clear(), "incomplete world"),
    (lambda rows: rows[0]["behavior"].pop("policy_entropy"), "incomplete behavior"),
    (lambda rows: rows[0]["world"].update(positive_reward_count=True), "scalar"),
    (lambda rows: rows[0]["world"].update(negative_reward_count=-1), "sample count"),
    (lambda rows: rows[0]["world"].update(zero_reward_count=1021), "B16/T64"),
    (lambda rows: rows[0]["world"].update(rewarded_count=1), "B16/T64"),
    (lambda rows: rows[0]["timing"].pop("world_train_seconds"), "incomplete timing"),
    (lambda rows: rows[0]["timing"].update(world_train_seconds=-0.1), "negative timing"),
    (lambda rows: rows[0]["timing"].update(total_seconds=0), "nonpositive"),
    (lambda rows: rows[0]["timing"].update(total_seconds=0.5), "exceed total"),
])
def test_broken_reports_are_rejected(tmp_path, mutation, message):
    rows = reports()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        audit_canary.audit(write_reports(tmp_path / "bad.jsonl", rows), 8)


@pytest.mark.parametrize("value", [None, True, "0", [], {}, float("nan"), float("inf"), 10**400])
def test_invalid_scalars_cannot_hide_in_metrics(tmp_path, value):
    rows = reports()
    rows[0]["behavior"]["policy_loss"] = value
    with pytest.raises(ValueError, match="scalar"):
        audit_canary.audit(write_reports(tmp_path / "bad.jsonl", rows), 8)


@pytest.mark.parametrize("value", [0, -1, True, 8.0])
def test_invalid_expected_update_budget(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        audit_canary.audit(tmp_path / "unused", value)


@pytest.mark.parametrize("mutation", [
    lambda text: text.rstrip("\n"),
    lambda text: "profiler startup\n" + text,
    lambda text: text.replace('"learner_step": 1,', '"learner_step": 0, "learner_step": 1,', 1),
    lambda text: text.replace('"total_loss": 0.0', '"total_loss": null, "total_loss": 0.0', 1),
    lambda text: text.replace("\n", "\n\n", 1),
])
def test_raw_output_must_be_complete_unambiguous_jsonl(tmp_path, mutation):
    path = write_reports(tmp_path / "bad.jsonl", reports())
    path.write_text(mutation(path.read_text()))
    with pytest.raises(ValueError):
        audit_canary.audit(path, 8)


@pytest.mark.parametrize("mutation, message", [
    (lambda timing: timing.pop("posterior_readback"), "transfer fields"),
    (lambda timing: timing["posterior_inputs"].update(calls=447), "transfer counts"),
    (lambda timing: timing["imagination_inputs"].update(bytes=1), "transfer counts"),
    (lambda timing: timing["posterior_readback"].update(calls=64.0), "transfer counts"),
    (lambda timing: timing["imagination_readback"].update(bytes=True), "scalar"),
    (lambda timing: timing["imagination_readback"].pop("wait_seconds"), "transfer fields"),
    (lambda timing: timing["posterior_readback"].update(wait_seconds=-1), "negative timing"),
    (lambda timing: timing.pop("posterior_sample_seconds"), "missing host"),
    (lambda timing: timing.update(imagination_targets_seconds=0), "nonpositive host"),
    (lambda timing: timing.update(imagination_targets_seconds=0.099), "exceed parent"),
    (lambda timing: timing["posterior_inputs"].update(seconds=0.099), "exceed parent"),
])
def test_profile_has_complete_bounded_measurements(tmp_path, mutation, message):
    rows = reports(True)
    mutation(rows[0]["timing"])
    with pytest.raises(ValueError, match=message):
        audit_canary.audit(write_reports(tmp_path / "bad.jsonl", rows), 8, readback_profile=True)


def test_profile_is_required_when_requested(tmp_path):
    path = write_reports(tmp_path / "plain.jsonl", reports())
    with pytest.raises(ValueError, match="transfer fields"):
        audit_canary.audit(path, 8, readback_profile=True)


def test_cli_exits_nonzero_for_empty_reports(tmp_path):
    path = write_reports(tmp_path / "empty.jsonl", [])
    result = subprocess.run([sys.executable, audit_canary.__file__, str(path), "--updates", "8"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "expected 8 reports, found 0" in result.stderr


@pytest.mark.parametrize("value", [{}, [], None, True])
def test_base_timings_must_be_scalars(tmp_path, value):
    rows = reports()
    rows[0]["timing"]["posterior_seconds"] = value
    with pytest.raises(ValueError, match="scalar"):
        audit_canary.audit(write_reports(tmp_path / "bad.jsonl", rows), 8)


def test_required_fields_track_the_rust_report_schema():
    root = Path(__file__).resolve().parents[2]
    source = (root / "kindle/src/dreamer/agent.rs").read_text()
    for section, struct in [("world", "WorldMetrics"), ("behavior", "BehaviorMetrics")]:
        body = source.split(f"pub struct {struct} {{", 1)[1].split("\n}", 1)[0]
        assert set(re.findall(r"pub (\w+):", body)) == audit_canary.METRICS[section]
    canary = (root / "kindle/examples/dreamer_canary.rs").read_text()
    assert "config.batch_size = 16;" in canary and "config.batch_length = 64;" in canary


def test_transfer_inventory_matches_the_declared_shape():
    batch, length, horizon, deter, stoch, actions, bins, observation = 16, 64, 15, 2048, 512, 18, 255, 3136
    starts, width = batch * length, deter + stoch
    assert audit_canary.TRANSFERS == {
        "posterior_inputs": (7 * length, 4 * starts * (2 * (width + actions) + observation)),
        "imagination_inputs": (4 * (horizon + 1) + 3 * horizon,
                               4 * starts * (3 * (horizon + 1) * width + horizon * (width + actions))),
        "posterior_readback": (length, 4 * starts * width),
        "imagination_readback": (2 * horizon + 1, 4 * starts * ((horizon + 1) * (actions + 3 * bins + 1) + horizon * width)),
    }


def test_cli_accepts_valid_reports_without_site_packages(tmp_path):
    path = write_reports(tmp_path / "reports.jsonl", reports(True))
    result = subprocess.run([sys.executable, "-I", "-S", audit_canary.__file__, str(path),
                             "--updates", "8", "--readback-profile"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["readback_report_fields_validated"]

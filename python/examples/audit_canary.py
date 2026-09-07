"""Validate complete, finite dreamer_canary reports; not tensors or gameplay.

The current canary fixes B16/T64 and fills 65 replay records once. The optional
readback-profile gate additionally fixes 12M/H15, matching its declared canary.
Validate process completion and checkpoint/provenance separately.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = {
    "world": set("""
        total_loss reconstruction_loss future_prediction_loss raw_kl dynamics_kl
        representation_kl reward_loss continuation_loss replay_value_loss
        replay_reward_prediction_mean replay_reward_target_mean replay_reward_mae
        rewarded_prediction_mean unrewarded_prediction_mean rewarded_count
        positive_reward_prediction_mean zero_reward_prediction_mean
        negative_reward_prediction_mean positive_reward_count zero_reward_count
        negative_reward_count replay_continuation_prediction_mean
        replay_continuation_target_mean replay_continuation_mae
    """.split()),
    "behavior": set("""
        total_loss policy_loss actor_update_scale value_loss replay_value_loss
        policy_entropy weighted_policy_entropy imagined_reward_mean
        imagined_continuation_mean return_mean return_scale advantage_abs_mean
        weighted_advantage_abs_mean
    """.split()),
}
STAGES = """
    replay_seconds posterior_seconds imagination_seconds world_train_seconds
    replay_refresh_seconds world_sync_seconds behavior_train_seconds behavior_sync_seconds
""".split()
READBACK_PHASES = "prepare_seconds submit_seconds wait_seconds copy_seconds".split()
HOST_PHASES = {
    "posterior": ["posterior_sample_seconds"],
    "imagination": """imagination_feature_seconds imagination_decode_seconds
        imagination_sample_seconds imagination_targets_seconds""".split(),
}
# Source inventory; staged native hardware tests must confirm these counts.
TRANSFERS = {
    "posterior_inputs": (448, 33_964_032),
    "imagination_inputs": (109, 661_708_800),
    "posterior_readback": (64, 10_485_760),
    "imagination_readback": (31, 208_666_624),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    require(type(value) in (int, float), "non-numeric scalar")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, "non-finite scalar")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_timing(timing, readback_profile):
    require(type(timing) is dict and set(STAGES + ["total_seconds"]) <= timing.keys(),
            "incomplete timing fields")
    for name, value in timing.items():
        if name in TRANSFERS:
            require(type(value) is dict, f"incomplete transfer fields: {name}")
            values = value.values()
        else:
            values = [value]
        for part in values:
            scalar(part)
            require(part >= 0, f"negative timing: {name}")
    require(timing["total_seconds"] > 0, "nonpositive total time")
    require(sum(timing[name] for name in STAGES) <= timing["total_seconds"] + 1e-9,
            "stage times exceed total")
    if not readback_profile:
        return
    for name, (calls, size) in TRANSFERS.items():
        record = timing.get(name)
        phases = READBACK_PHASES if name.endswith("readback") else ["seconds"]
        require(type(record) is dict and set(record) == set(["calls", "bytes", *phases]),
                f"incomplete transfer fields: {name}")
        require(type(record["calls"]) is int and record["calls"] == calls
                and type(record["bytes"]) is int and record["bytes"] == size,
                f"wrong 12M/B16/T64/H15 transfer counts: {name}")
    for phase, names in HOST_PHASES.items():
        require(all(name in timing for name in names), f"missing host timing: {phase}")
        require(all(timing[name] > 0 for name in names), f"nonpositive host timing: {phase}")
        elapsed = sum(timing[f"{phase}_readback"][name] for name in READBACK_PHASES)
        elapsed += timing[f"{phase}_inputs"]["seconds"] + sum(timing[name] for name in names)
        require(elapsed <= timing[f"{phase}_seconds"] + 1e-9,
                f"substage times exceed parent: {phase}")


def audit(path, updates, *, readback_profile=False):
    require(type(updates) is int and updates > 0, "expected updates must be a positive integer")
    data = Path(path).read_bytes()
    lines = data.decode("utf-8").splitlines(keepends=True)
    require(len(lines) == updates, f"expected {updates} reports, found {len(lines)}")
    for step, line in enumerate(lines, 1):
        require(line.endswith("\n"), f"report {step}: incomplete line")
        report = json.loads(line, object_pairs_hook=unique_object)
        require(type(report) is dict
                and set(report) == {"learner_step", "replay_len", "world", "behavior", "timing"},
                f"report {step}: not a complete LearnReport")
        require(type(report["learner_step"]) is int and report["learner_step"] == step,
                f"report {step}: not a fresh contiguous learner sequence")
        require(type(report["replay_len"]) is int and report["replay_len"] == 65,
                f"report {step}: expected the canary's fixed 65-record replay")
        for section, fields in METRICS.items():
            metrics = report[section]
            require(type(metrics) is dict and fields <= metrics.keys(), f"incomplete {section} metrics")
            for value in metrics.values():
                scalar(value)
        world = report["world"]
        for name in ("rewarded_count", "positive_reward_count", "negative_reward_count", "zero_reward_count"):
            require(type(world[name]) is int and world[name] >= 0, f"invalid sample count: {name}")
        require(world["positive_reward_count"] + world["negative_reward_count"] == world["rewarded_count"]
                and world["rewarded_count"] + world["zero_reward_count"] == 1024,
                "reward sample counts do not cover B16/T64")
        validate_timing(report["timing"], readback_profile)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), reports=updates,
                reports_valid=True, readback_report_fields_validated=readback_profile,
                scope="report validation only; not tensor parity, runtime provenance, or gameplay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--readback-profile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.reports, args.updates, readback_profile=args.readback_profile), indent=2))


if __name__ == "__main__":
    main()

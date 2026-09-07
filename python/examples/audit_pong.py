"""Audit final-checkpoint Pong mastery, not the historical Atari-100k window.

The three-seed LeVJEPA gate is declared in
docs/experiments/2026-09-06-levjepa-pong.md. This reads artifacts only; it never
trains, chooses checkpoints, drops timeouts or changes the acceptance threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from kindle._atari_scores import _read_events
from kindle._vector_audit import audit as audit_vector_accounting


TRAIN_STEPS = 200_000
EVALUATION_STEPS = 75_000
CONTROL_STEPS = 20_000
ENCODER_SHA256 = "da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value, *, numeric: bool = False) -> None:
    if numeric:
        require(type(value) in (int, float, dict, list), "non-numeric learner report value")
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite report value")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child, numeric=numeric)
    elif isinstance(value, list):
        for child in value:
            finite(child, numeric=numeric)


def audit_run(path: Path) -> dict:
    start = end = None
    checkpoint = None
    episodes = []
    intervals = []
    updates = 0
    first_update = None
    update_seconds = 0.0
    for event in _read_events(path):
        finite(event)
        kind = event["event"]
        require(end is None, f"{path}: event after run_end")
        if kind == "run_start":
            require(start is None, f"{path}: multiple run segments")
            require(event.get("environment") == "ALE/Pong-v5", f"{path}: Pong-only scorer")
            start = event
            continue
        require(start is not None, f"{path}: missing run_start")
        if kind == "learner":
            updates += 1
            report = event["report"]
            # serde_json encodes non-finite Rust floats as null. Learner
            # metrics have no optional leaves; null must not hide a NaN.
            finite(report, numeric=True)
            require(report["learner_step"] == start["starting_learner_step"] + updates,
                    f"{path}: learner counter gap")
            if first_update is None:
                first_update = event["run_step"]
            require(event["run_step"] == first_update + 4 * (updates - 1),
                    f"{path}: ratio-256 update schedule changed")
            update_seconds += report["timing"]["total_seconds"]
        elif kind == "episode":
            previous_step = episodes[-1]["run_step"] if episodes else 0
            require(event["episode"] == len(episodes) + 1
                    and event["run_step"] - previous_step == event["length"],
                    f"{path}: episode counter/length gap")
            require(type(event["terminated"]) is bool and type(event["truncated"]) is bool
                    and (event["terminated"] or event["truncated"]),
                    f"{path}: incomplete episode reported as complete")
            episodes.append(event)
        elif kind == "interval":
            previous_step = intervals[-1]["run_step"] if intervals else 0
            require(event["run_step"] - previous_step == event["interval_steps"],
                    f"{path}: interval gap")
            require(len(event["action_counts"]) == 18
                    and sum(event["action_counts"]) == event["interval_steps"],
                    f"{path}: interval action ledger mismatch")
            intervals.append(event)
        elif kind == "run_end":
            end = event
        elif kind == "checkpoint":
            checkpoint = event
    require(start is not None and end is not None, f"{path}: incomplete run")
    steps = start["steps"]
    for key in ("environment", "seed", "mode", "steps"):
        require(start[key] == end[key], f"{path}: changed {key}")
    require(end["environment_step_delta"] == steps
            and end["environment_step"] == start["starting_environment_step"] + steps,
            f"{path}: interaction counter mismatch")
    require(intervals and intervals[-1]["run_step"] == steps,
            f"{path}: incomplete interval ledger")
    require(end["learner_updates"] == end["learner_step_delta"] == updates
            == sum(item["learner_updates"] for item in intervals),
            f"{path}: update ledger mismatch")
    require(end["action_counts"] == [sum(item["action_counts"][i] for item in intervals)
                                     for i in range(18)]
            and sum(end["action_counts"]) == steps,
            f"{path}: action ledger mismatch")
    for interval_key, end_key in (("reward", "total_reward"),
                                  ("intrinsic_reward", "total_intrinsic_reward"),
                                  ("positive_reward_events", "positive_reward_events"),
                                  ("negative_reward_events", "negative_reward_events"),
                                  ("completed_episodes", "completed_episodes")):
        require(sum(item[interval_key] for item in intervals) == end[end_key],
                f"{path}: {end_key} ledger mismatch")
    require(end["total_reward"] == end["positive_reward_events"] - end["negative_reward_events"]
            == sum(item["return"] for item in episodes) + end["partial_episode_return"],
            f"{path}: Pong point ledger mismatch")
    require(0 <= end["positive_reward_events"] + end["negative_reward_events"] <= steps
            and all(type(count) is int and count >= 0 for count in end["action_counts"]),
            f"{path}: impossible event/action counts")
    require(sum(item["length"] for item in episodes) + end["partial_episode_length"] == steps
            and len(episodes) == end["completed_episodes"],
            f"{path}: episode ledger mismatch")
    require(end["reset_noop_frames"] == 0 and end["emulator_resets"] == len(episodes) + 1
            and steps <= end["executed_action_frames"] <= 4 * steps,
            f"{path}: emulator clock/reset mismatch")
    if updates:
        # Initial frame plus reset records from strictly earlier actions count
        # toward the 1,088-row replay warmup. Resets happen after learning.
        expected_first = next(step for step in range(1, 1088)
                              if step + 1 + sum(ep["run_step"] < step for ep in episodes) >= 1088)
        require(first_update == expected_first and updates == 1 + (steps - first_update) // 4,
                f"{path}: warmup or final update count mismatch")
    natural = [ep for ep in episodes if ep["terminated"] and not ep["truncated"]]
    wins = sum(ep["return"] > 0 for ep in natural)
    mean = sum(ep["return"] for ep in episodes) / len(episodes) if episodes else 0.0
    require(math.isclose(mean, end["mean_completed_return"], abs_tol=1e-9),
            f"{path}: mean score mismatch")
    return {"path": str(path), "sha256": sha256(path), "start": start, "end": end,
            "checkpoint": checkpoint,
            "natural_games": len(natural), "timeouts": sum(ep["truncated"] for ep in episodes),
            "natural_wins": wins, "win_fraction": wins / len(episodes) if episodes else 0.0,
            "mean_return": mean, "learner_seconds": update_seconds,
            "simulated_to_wall": end["executed_action_frames"] / 60 / end["elapsed_seconds"]}


def audit_vector_run(path: Path) -> dict:
    accounting = audit_vector_accounting(path)
    require(accounting["budget_complete"], f"{path}: interrupted vector budget")
    start = end = checkpoint = None
    episodes = []
    update_seconds = 0.0
    for event in _read_events(path):
        finite(event)
        kind = event["event"]
        if kind == "run_start":
            require(event.get("environment") == "ALE/Pong-v5", f"{path}: Pong-only scorer")
            # Normalize only field names for the shared checkpoint/recipe audit.
            start = {**event, "perception": event["model_provenance"]["perception"],
                     "trainable_parameters": event["trainable_parameter_counts"]}
        elif kind == "run_end":
            end = event
        elif kind == "checkpoint":
            checkpoint = event
        elif kind == "episode":
            episodes.append(event)
        elif kind == "learner":
            update_seconds += event["report"]["timing"]["total_seconds"]
        elif kind == "transition":
            require(all(reward in (-1, 0, 1) for reward in event["rewards"]),
                    f"{path}: not an unshaped Pong point reward")
            require(all(pair[1] == 0 for pair in event["stored_rewards"]),
                    f"{path}: intrinsic reward in extrinsic-only experiment")
    require(end["reset_noop_frames"] == [0] * start["num_envs"]
            and end["emulator_resets"] == [count + 1 for count in end["episode_counts"]],
            f"{path}: emulator clock/reset mismatch")
    natural = sum(ep["terminated"] and not ep["truncated"] for ep in episodes)
    wins = sum(ep["terminated"] and not ep["truncated"] and ep["episode_return"] > 0 for ep in episodes)
    return {"path": str(path), "sha256": sha256(path), "start": start, "end": end,
            "checkpoint": checkpoint, "natural_games": natural,
            "timeouts": sum(ep["truncated"] for ep in episodes), "natural_wins": wins,
            "win_fraction": wins / len(episodes) if episodes else 0.0,
            "mean_return": accounting["mean_completed_return"] if episodes else 0.0,
            "learner_seconds": update_seconds,
            "simulated_to_wall": end["aggregate_simulated_wall_ratio"],
            "per_stream_simulated_to_wall": end["per_stream_simulated_wall_ratio"]}


def validate_protocol(run: dict, mode: str, steps: int, vector_envs: int = 0) -> None:
    start = run["start"]
    expected = {"environment": "ALE/Pong-v5", "ale_py_version": "0.12.1",
                "atari_protocol": "published", "action_repeat": 4,
                "full_action_space": True, "noop_max": 0, "max_episode_frames": 100_000,
                "mode": mode, "steps": steps}
    if vector_envs:
        expected.update(protocol="kindle-vector-v1", num_envs=vector_envs, sticky_actions=0.0,
                        policy_seed_rule="config.seed + stream (wrapping u64)")
        require(start["environment_seeds"] == [(start["seed"] + stream * 1_000_003) % 2**32
                                              for stream in range(vector_envs)], "changed environment seed rule")
        require(start["config"]["action_count"] == len(start["action_meanings"]) == 18,
                "changed action vocabulary")
    else:
        expected.update(actions=18, random_action_steps=0, output_appended=False)
    for key, value in expected.items():
        require(start[key] == value, f"{run['path']}: expected {key}={value!r}")
    identity = start["perception"]
    require(identity["kind"] == "levjepa" and identity["checkpoint_sha256"] == ENCODER_SHA256
            and identity["model_id"] == "galilai-group/LeVJEPA-VideoMix-Large"
            and identity["checkpoint_revision"] == "e831a0347737fcaa660b39c57d41c109de399845"
            and identity["encoding_revision"] == "levjepa-large-f32-chunk16-letterbox224-jl64-pool2-v1"
            and start["model_provenance"]["perception"] == identity,
            f"{run['path']}: not the pinned LeVJEPA experiment")
    if not vector_envs:
        require(start["encoder_checkpoint_sha256"] == ENCODER_SHA256
                and run["end"]["total_intrinsic_reward"] == 0, f"{run['path']}: encoder or reward mismatch")
    require(start["config"]["seed"] == start["seed"], f"{run['path']}: seed mismatch")


def audit_checkpoint(path: Path, train: dict, evaluation: dict) -> dict:
    metadata = json.loads((path / "metadata.json").read_text())
    require(metadata["format"] == 3 and metadata["architecture"] == "dreamerv3-visual-features",
            f"{path}: unsupported checkpoint")
    if "num_envs" in train["start"]:
        require(metadata["collection_streams"] == train["start"]["num_envs"],
                f"{path}: wrong checkpoint collection stream count")
    for key in ("config", "perception"):
        require(metadata[key] == train["start"][key] == evaluation["start"][key],
                f"{path}: changed {key}")
    require(metadata["environment_step"] == TRAIN_STEPS == evaluation["start"]["starting_environment_step"]
            and metadata["learner_step"] == train["end"]["learner_updates"]
            == evaluation["start"]["starting_learner_step"], f"{path}: not the final checkpoint")
    restored = evaluation["start"]["restored_checkpoint"]
    saved = train["checkpoint"]
    require(saved is not None and saved["run_step"] == TRAIN_STEPS
            and saved["learner_step"] == metadata["learner_step"], f"{path}: missing final save event")
    require(restored["metadata_sha256"] == sha256(path / "metadata.json")
            == saved["identity"]["metadata_sha256"]
            and restored["tensor_sha256"] == metadata["tensor_sha256"]
            == saved["identity"]["tensor_sha256"],
            f"{path}: evaluation loaded different checkpoint files")
    for key in ("dreamerv3_revision", "meganeura_revision", "blade_revision", "future_head_revision"):
        require(metadata[key] == train["start"]["model_provenance"][key], f"{path}: changed {key}")
    tensors = {}
    for name in ("world.safetensors", "behavior.safetensors", "slow_value.safetensors"):
        digest = sha256(path / name)
        require(metadata["tensor_sha256"][Path(name).stem] == digest, f"{path}: damaged {name}")
        with safe_open(path / name, framework="numpy") as model:
            require(len(model.keys()) > 0, f"{path}: empty {name}")
            for key in model.keys():
                require(np.isfinite(model.get_tensor(key)).all(), f"{path}: non-finite {name}:{key}")
            tensors[name] = {"sha256": digest, "tensors": len(model.keys())}
    return {"path": str(path), "metadata_sha256": sha256(path / "metadata.json"), "files": tensors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs=3, required=True)
    parser.add_argument("--evaluation", type=Path, nargs=3, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs=3, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vector-envs", type=int, choices=(2, 4, 8),
                        help="new vector protocol: this many training streams; one frozen evaluation/control stream")
    args = parser.parse_args()
    read_run = audit_vector_run if args.vector_envs else audit_run
    training = [read_run(path) for path in args.train]
    evaluations = [read_run(path) for path in args.evaluation]
    require([run["start"]["seed"] for run in training] == [0, 1, 2]
            == [run["start"]["seed"] for run in evaluations], "supply independent seeds 0, 1, 2 in order")
    results = []
    identity_keys = ("perception", "model_provenance", "native_extension_sha256", "runner_sha256",
                     "trainable_parameters", "action_meanings")
    if args.vector_envs:
        identity_keys += ("wrapper_sha256", "cpu_worker_threads")
    reference_config = dict(training[0]["start"]["config"])
    reference_config.pop("seed")
    for key, value in {"model_size": "size12_m", "batch_size": 16, "batch_length": 64,
                       "world_backprop_length": 64, "world_microbatch_size": 16,
                       "train_ratio": 256, "learning_rate": 4e-5, "behavior_learning_rate": None,
                       "learning_rate_warmup": 1000, "agc": 0.3, "extrinsic_reward_scale": 1.0,
                       "intrinsic_reward_scale": 0.0, "visitation_bonus": False}.items():
        require(reference_config[key] == value, f"recipe changed {key}")
    require(reference_config["loss_scales"]["reconstruction"] == 0
            and reference_config["loss_scales"]["future_prediction"] == 0.25, "objective changed")
    for train, evaluation, checkpoint in zip(training, evaluations, args.checkpoint):
        validate_protocol(train, "train", TRAIN_STEPS, args.vector_envs or 0)
        validate_protocol(evaluation, "evaluate_sample", EVALUATION_STEPS, 1 if args.vector_envs else 0)
        require(train["start"]["starting_environment_step"] == train["start"]["starting_learner_step"] == 0
                and train["start"]["restored_checkpoint"] is None
                and train["end"]["learner_updates"] > 0 and evaluation["end"]["learner_updates"] == 0,
                "training must be fresh and evaluation frozen")
        config = dict(train["start"]["config"])
        config.pop("seed")
        require(config == reference_config, "different training recipes across seeds")
        for key in identity_keys:
            require(train["start"][key] == evaluation["start"][key] == training[0]["start"][key],
                    f"changed implementation identity: {key}")
        checkpoint_result = audit_checkpoint(checkpoint, train, evaluation)
        passed = (evaluation["natural_games"] >= 20 and evaluation["mean_return"] >= 15
                  and evaluation["win_fraction"] >= 0.90)
        results.append({"seed": train["start"]["seed"], "passed": passed,
                        "training": train, "evaluation": evaluation, "checkpoint": checkpoint_result})
    control = read_run(args.control)
    validate_protocol(control, "train", CONTROL_STEPS, 1 if args.vector_envs else 0)
    require(control["start"]["starting_environment_step"] == control["start"]["starting_learner_step"]
            == control["end"]["learner_updates"] == 0 and control["start"]["seed"] == 0,
            "control must be a fresh zero-update seed-0 policy")
    require(control["start"]["restored_checkpoint"] is None, "control must not restore trained weights")
    control_config = dict(control["start"]["config"])
    control_config.pop("seed")
    require(control_config.pop("train_ratio") == 0, "control must disable scheduled learning")
    require(control_config == {k: v for k, v in reference_config.items() if k != "train_ratio"},
            "control changed more than the train ratio")
    for key in identity_keys:
        require(control["start"][key] == training[0]["start"][key], f"control changed {key}")
    output = {"passed": all(item["passed"] for item in results), "seeds": results, "control": control,
              "protocol": "kindle-vector-v1" if args.vector_envs else "serial",
              "training_streams": args.vector_envs or 1}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": output["passed"], "output": str(args.output)}))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

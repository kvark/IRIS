"""Measure open-loop Atari dynamics in frozen visual feature space.

Replay a recorded frozen match or follow deterministic forced-random actions.
Forecasts consume only the origin belief and proposed controls, never the target
frames. Recorded future controls are retrospective conditioning, not a live plan.
Features are not RGB reconstructions; prior reward forecasts are distinct from
posterior reward inference after seeing the frame.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from kindle._reward_probe import RewardProbe
import atari
from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing, sha256_file


ACTION_SEED_XOR = 0xA7A2_1000
CONTROL_ACTION_SEED_XOR = 0xD1A6_4057


def checked_mean(total, count):
    return total / count if count else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recorded_first_game(path):
    """Read one complete frozen N=1 game, retaining its exact source prefix."""
    prefix = hashlib.sha256()
    byte_count = 0
    header, transitions, episode = None, [], None
    with path.open("rb") as source:
        for line in source:
            require(line.endswith(b"\n"), "incomplete recorded game")
            prefix.update(line)
            byte_count += len(line)
            row = json.loads(line)
            if header is None:
                header = row
                require(header.get("event") == "run_start", "missing run header")
                require(
                    header.get("protocol") == "kindle-vector-v1"
                    and header.get("num_envs") == 1
                    and header.get("mode") == "evaluate_sample",
                    "requires frozen N=1 evaluation",
                )
                require(
                    header["environment_seeds"] == [header["seed"]],
                    "changed stream seed",
                )
            elif row["event"] == "transition":
                require(
                    not transitions
                    or not (
                        transitions[-1]["terminated"][0]
                        or transitions[-1]["truncated"][0]
                    ),
                    "transition after episode boundary",
                )
                require(
                    row["run_step"] == row["vector_tick"] == len(transitions) + 1,
                    "non-contiguous recorded actions",
                )
                for field in (
                    "actions",
                    "rewards",
                    "stored_rewards",
                    "terminated",
                    "truncated",
                    "executed_action_frames",
                ):
                    require(len(row[field]) == 1, "recorded stream count differs")
                transitions.append(row)
            elif row["event"] == "progress":
                require(
                    row["run_step"] == len(transitions)
                    and row["learner_step"] == header["starting_learner_step"],
                    "recorded counters changed",
                )
            elif row["event"] == "episode":
                episode = row
                break
            else:
                raise ValueError("unexpected event in frozen game")
    require(episode is not None and transitions, "no complete recorded game")
    require(episode["stream"] == episode["episode"] == 0, "not the first game")
    require(
        episode["episode_length"]
        == episode["run_step"]
        == episode["stream_step"]
        == len(transitions),
        "recorded length differs",
    )
    require(
        episode["episode_return"] == sum(row["rewards"][0] for row in transitions),
        "recorded return differs",
    )
    require(
        episode["terminated"] == transitions[-1]["terminated"][0]
        and episode["truncated"] == transitions[-1]["truncated"][0]
        and (episode["terminated"] or episode["truncated"]),
        "recorded boundary differs",
    )
    return (
        header,
        transitions,
        {
            "path": str(path.resolve()),
            "prefix_sha256": prefix.hexdigest(),
            "prefix_bytes": byte_count,
            "episode": episode,
        },
    )


def verify_transition(row, action, reward, terminated, truncated, frames):
    expected = dict(
        actions=[action],
        rewards=[reward],
        stored_rewards=[[reward, 0.0]],
        terminated=[terminated],
        truncated=[truncated],
        executed_action_frames=[frames],
    )
    require(
        all(row[key] == value for key, value in expected.items()),
        f"recorded transition differs at action {row['run_step']}",
    )


def feature_mse(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    require(
        prediction.shape == target.shape and prediction.size > 0,
        "feature prediction shape differs",
    )
    require(
        np.isfinite(prediction).all() and np.isfinite(target).all(),
        "nonfinite features",
    )
    return float(np.mean(np.square(prediction - target)))


def horizon_reward_summary(probe):
    # The shared helper also serves one-step live evaluation. Here the enclosing
    # horizon supplies the forecast length, so do not label every prior one-step.
    result = {
        key.removeprefix("one_step_"): value for key, value in probe.summary().items()
    }
    result["by_reward_sign"] = {
        sign: {key.removeprefix("one_step_"): value for key, value in stats.items()}
        for sign, stats in result["by_reward_sign"].items()
    }
    return result


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder_checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("environment", nargs="?")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--atari-protocol", choices=tuple(ATARI_PROTOCOLS))
    parser.add_argument("--output")
    parser.add_argument(
        "--trace", help="Fresh JSONL path for per-target prediction errors"
    )
    parser.add_argument(
        "--recorded-run", type=Path, help="Replay its first completed frozen N=1 game"
    )
    args = parser.parse_args()
    source_header = source_rows = source_record = None
    if args.recorded_run:
        if (
            args.steps is not None
            or args.seed is not None
            or args.environment
            or args.atari_protocol
        ):
            parser.error(
                "--recorded-run supplies the budget, seed, environment and protocol"
            )
        source_header, source_rows, source_record = recorded_first_game(
            args.recorded_run
        )
        args.steps, args.seed = len(source_rows), source_header["seed"]
        args.environment, args.atari_protocol = (
            source_header["environment"],
            source_header["atari_protocol"],
        )
        require(
            source_header["ale_py_version"] == ale_py.__version__, "ALE version differs"
        )
        require(
            source_header["wrapper_sha256"] == sha256_file(atari.__file__),
            "wrapper differs",
        )
    else:
        args.steps = 5_000 if args.steps is None else args.steps
        args.seed = 100 if args.seed is None else args.seed
        args.environment = args.environment or "ALE/Pong-v5"
        args.atari_protocol = args.atari_protocol or "published"
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.horizon <= 0 or args.horizon > args.steps:
        parser.error("--horizon must be in [1, steps]")
    if args.stride <= 0:
        parser.error("--stride must be positive")

    protocol = ATARI_PROTOCOLS[args.atari_protocol]
    if source_header:
        expected = dict(
            action_repeat=4,
            noop_max=protocol.noop_max,
            max_episode_frames=protocol.max_episode_frames,
            full_action_space=protocol.full_action_space,
            sticky_actions=0.0,
        )
        require(
            all(source_header.get(key) == value for key, value in expected.items()),
            "recorded environment protocol differs",
        )
    for path in (args.output, args.trace):
        if path:
            require(
                not Path(path).exists() and Path(path).parent.is_dir(),
                "output must be fresh",
            )
    if args.output and args.trace:
        require(
            Path(args.output).resolve() != Path(args.trace).resolve(),
            "output and trace must differ",
        )
    environment = gym.make(
        args.environment,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=protocol.full_action_space,
    )
    environment = DreamerAtariPreprocessing(
        environment,
        noop_max=protocol.noop_max,
        max_episode_frames=protocol.max_episode_frames,
    )
    frame, _ = environment.reset(seed=args.seed)
    action_count = int(environment.action_space.n)
    agent = kindle.Agent.restore(args.checkpoint, args.encoder_checkpoint)
    if (
        agent.config["intrinsic_reward_scale"] != 0
        or agent.config["extrinsic_reward_scale"] != 1
    ):
        raise ValueError(
            "this probe's reward labels require unscaled extrinsic-only training"
        )
    if int(agent.config["action_count"]) != action_count:
        raise ValueError("checkpoint and environment action counts differ")
    agent.begin_episode(frame)
    starting_environment_step = agent.environment_step
    starting_learner_step = agent.learner_step
    import kindle._native as native

    native_sha256 = sha256_file(native.__file__)
    checkpoint = Path(args.checkpoint)
    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in [
            checkpoint / "metadata.json",
            *sorted(checkpoint.glob("*.safetensors")),
        ]
    }
    if source_header:
        require(
            source_header["native_extension_sha256"] == native_sha256,
            "use the recorded native executable",
        )
        require(
            agent.config == source_header["config"]
            and agent.provenance == source_header["model_provenance"],
            "recorded model identity differs",
        )
        require(
            starting_environment_step == source_header["starting_environment_step"]
            and starting_learner_step == source_header["starting_learner_step"],
            "restored counters differ",
        )
        restored = source_header["restored_checkpoint"]
        require(
            checkpoint_hashes["metadata.json"] == restored["metadata_sha256"],
            "checkpoint metadata differs",
        )
        for name, expected in restored["tensor_sha256"].items():
            require(
                checkpoint_hashes[f"{name}.safetensors"] == expected,
                "checkpoint tensor differs",
            )
        require(
            list(environment.action_meanings) == source_header["action_meanings"],
            "action vocabulary differs",
        )

    action_rng = random.Random(args.seed ^ ACTION_SEED_XOR)
    actions = (
        [row["actions"][0] for row in source_rows]
        if source_rows
        else [action_rng.randrange(action_count) for _ in range(args.steps)]
    )
    control_action_rng = random.Random(args.seed ^ CONTROL_ACTION_SEED_XOR)
    control_actions = [
        control_action_rng.randrange(action_count) for _ in range(args.steps)
    ]
    horizon = args.horizon
    model_mse_sum = [0.0] * horizon
    control_mse_sum = [0.0] * horizon
    persistence_mse_sum = [0.0] * horizon
    reward_mae_sum = [0.0] * horizon
    control_reward_mae_sum = [0.0] * horizon
    sample_count = [0] * horizon
    reward_probes = [RewardProbe() for _ in range(horizon)]
    continuation_mse_sum = [0.0] * horizon
    continue_baseline_mse_sum = [0.0] * horizon
    terminal_count = [0] * horizon
    truncated_count = [0] * horizon
    discount = float(
        np.float32(1.0) - np.float32(1.0) / np.float32(agent.config["horizon"])
    )
    posterior_mse_sum = 0.0
    posterior_count = 0
    rollout_starts = 0
    episodes = 0
    pending = {}
    is_first = True
    forecasts = agent.config["loss_scales"]["future_prediction"] > 0

    with ExitStack() as stack:
        stack.callback(environment.close)
        trace = (
            stack.enter_context(open(args.trace, "x", encoding="utf-8"))
            if args.trace
            else None
        )
        for step, action in enumerate(actions):
            if source_rows:
                require(
                    agent.act() == action,
                    f"sampled policy diverged at action {step + 1}",
                )
            else:
                action_mask = [index == action for index in range(action_count)]
                require(
                    agent.act(action_mask=action_mask) == action,
                    "forced action was not honored",
                )
            length = min(horizon, args.steps - step) if step % args.stride == 0 else 1
            if length:
                start_observation = np.asarray(
                    agent.visual_observation, dtype=np.float64
                )
                posterior = np.asarray(agent.observation_prediction(), dtype=np.float64)
                if not is_first or not forecasts:
                    posterior_mse_sum += float(
                        np.mean(np.square(posterior - start_observation))
                    )
                    posterior_count += 1
                predicted_rewards, predicted_observations = (
                    agent.prior_diagnostic_rollout(actions[step : step + length])
                )
                control_rewards, control_observations = agent.prior_diagnostic_rollout(
                    control_actions[step : step + length]
                )
                behavior_rewards, continuations, _ = agent.prior_behavior_rollout(
                    actions[step : step + length]
                )
                require(
                    predicted_rewards == behavior_rewards,
                    "diagnostic RNG or rewards differ",
                )
                if (
                    len(predicted_rewards) != length
                    or len(predicted_observations) != length
                    or len(continuations) != length
                ):
                    raise RuntimeError("prior rollout returned the wrong horizon")
                if (
                    len(control_rewards) != length
                    or len(control_observations) != length
                ):
                    raise RuntimeError("control rollout returned the wrong horizon")
                predictions = zip(
                    predicted_rewards,
                    predicted_observations,
                    control_rewards,
                    control_observations,
                    continuations,
                )
                for offset, prediction in enumerate(predictions):
                    (
                        reward,
                        observation,
                        control_reward,
                        control_observation,
                        continuation,
                    ) = prediction
                    target_step = step + offset
                    pending.setdefault(target_step, []).append(
                        (
                            offset,
                            float(reward),
                            np.asarray(observation, dtype=np.float64),
                            float(control_reward),
                            np.asarray(control_observation, dtype=np.float64),
                            start_observation,
                            float(continuation),
                        )
                    )
                rollout_starts += 1

            frame, reward, terminated, truncated, _ = environment.step(action)
            reward = float(reward)
            terminated = bool(terminated)
            truncated = bool(truncated)
            if source_rows:
                verify_transition(
                    source_rows[step],
                    action,
                    reward,
                    terminated,
                    truncated,
                    environment.executed_action_frames,
                )
            agent.observe(
                frame,
                extrinsic_reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            target_observation = np.asarray(agent.visual_observation, dtype=np.float64)
            posterior_reward = float(agent.posterior_reward_prediction())
            continuation_target = 0.0 if terminated else discount
            for (
                offset,
                predicted_reward,
                predicted_observation,
                control_reward,
                control_observation,
                start_observation,
                predicted_continuation,
            ) in pending.pop(step, []):
                model_error = feature_mse(predicted_observation, target_observation)
                control_error = feature_mse(control_observation, target_observation)
                persistence_error = feature_mse(start_observation, target_observation)
                model_mse_sum[offset] += model_error
                control_mse_sum[offset] += control_error
                persistence_mse_sum[offset] += persistence_error
                reward_mae_sum[offset] += abs(predicted_reward - reward)
                control_reward_mae_sum[offset] += abs(control_reward - reward)
                sample_count[offset] += 1
                reward_probes[offset].record(reward, predicted_reward, posterior_reward)
                require(
                    np.isfinite(predicted_continuation)
                    and 0 <= predicted_continuation <= 1,
                    "invalid continuation probability",
                )
                continuation_mse_sum[offset] += (
                    predicted_continuation - continuation_target
                ) ** 2
                continue_baseline_mse_sum[offset] += (
                    discount - continuation_target
                ) ** 2
                terminal_count[offset] += int(terminated)
                truncated_count[offset] += int(truncated)
                if trace:
                    print(
                        json.dumps(
                            dict(
                                target_action=step + 1,
                                horizon=offset + 1,
                                origin_action=step - offset,
                                action=action,
                                reward=reward,
                                prior_reward=predicted_reward,
                                posterior_reward=posterior_reward,
                                unrelated_action_reward=control_reward,
                                continuation=predicted_continuation,
                                continuation_target=continuation_target,
                                terminated=terminated,
                                truncated=truncated,
                                feature_mse=model_error,
                                persistence_mse=persistence_error,
                                unrelated_action_feature_mse=control_error,
                            ),
                            allow_nan=False,
                        ),
                        file=trace,
                    )

            if terminated or truncated:
                episodes += 1
                pending.clear()
                if step + 1 < args.steps:
                    frame, _ = environment.reset()
                    agent.begin_episode(frame)
            is_first = terminated or truncated
    require(
        agent.learner_step == starting_learner_step, "diagnostic performed learning"
    )
    require(
        agent.environment_step - starting_environment_step == args.steps,
        "interaction count differs",
    )
    require(not pending, "unresolved forecast targets")

    model_mse = [
        checked_mean(total, count) for total, count in zip(model_mse_sum, sample_count)
    ]
    control_mse = [
        checked_mean(total, count)
        for total, count in zip(control_mse_sum, sample_count)
    ]
    persistence_mse = [
        checked_mean(total, count)
        for total, count in zip(persistence_mse_sum, sample_count)
    ]
    result = {
        "protocol": "kindle-world-probe-v2",
        "source": "recorded_frozen_policy" if source_rows else "forced_random_coverage",
        "recorded_game": source_record,
        "sampled_actions_match_source": True if source_rows else None,
        "native_extension_sha256": native_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "encoder_sha256": sha256_file(args.encoder_checkpoint),
        "script_sha256": sha256_file(__file__),
        "reward_probe_sha256": sha256_file(
            Path(kindle.__file__).with_name("_reward_probe.py")
        ),
        "wrapper_sha256": sha256_file(atari.__file__),
        "model_provenance": agent.provenance,
        "learner_updates": agent.learner_step - starting_learner_step,
        "forecast_sampling": "one reproducible diagnostic latent draw per step; live RNG unchanged",
        "one_step_every_action": True,
        "trace_sha256": sha256_file(args.trace) if args.trace else None,
        "environment": args.environment,
        "atari_protocol": args.atari_protocol,
        "checkpoint": args.checkpoint,
        "starting_environment_step": starting_environment_step,
        "steps": args.steps,
        "seed": args.seed,
        "horizon": horizon,
        "stride": args.stride,
        "rollout_starts": rollout_starts,
        "completed_episodes": episodes,
        "gpu_device": agent.gpu_device,
        "observation_prediction_source": (
            "deterministic_forecast" if forecasts else "posterior_reconstruction"
        ),
        "observed_state_prediction_count": posterior_count,
        "observed_state_prediction_mse": checked_mean(
            posterior_mse_sum, posterior_count
        ),
        "sample_count_by_horizon": sample_count,
        "prior_observation_mse_by_horizon": model_mse,
        "unrelated_action_mse_by_horizon": control_mse,
        "prior_over_unrelated_action_by_horizon": [
            model / control
            if model is not None and control is not None and control > 0.0
            else None
            for model, control in zip(model_mse, control_mse)
        ],
        "persistence_mse_by_horizon": persistence_mse,
        "prior_over_persistence_by_horizon": [
            model / baseline
            if model is not None and baseline is not None and baseline > 0.0
            else None
            for model, baseline in zip(model_mse, persistence_mse)
        ],
        "prior_reward_mae_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(reward_mae_sum, sample_count)
        ],
        "unrelated_action_reward_mae_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(control_reward_mae_sum, sample_count)
        ],
        "reward_calibration_by_horizon": [
            horizon_reward_summary(probe) for probe in reward_probes
        ],
        "continuation_target": "0 for terminated, otherwise f32(1 - 1 / config.horizon); truncation is not terminal",
        "continuation_mse_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(continuation_mse_sum, sample_count)
        ],
        "always_continue_mse_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(continue_baseline_mse_sum, sample_count)
        ],
        "terminal_count_by_horizon": terminal_count,
        "truncation_count_by_horizon": truncated_count,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as output:
            output.write(encoded)
            output.write("\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()

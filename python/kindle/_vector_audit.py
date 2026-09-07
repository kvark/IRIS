"""CPU-only vector accounting with descriptive episode counts, not game wins."""

from collections import deque
import json
import math
from pathlib import Path
import struct


VECTOR_PROTOCOL = "kindle-vector-v2"


def episode_summary(episodes):
    for episode in episodes:
        if type(episode["episode_return"]) not in (int, float):
            raise ValueError("episode return must be a finite scalar")
        require_numbers(episode["episode_return"])
        if (type(episode["terminated"]) is not bool or type(episode["truncated"]) is not bool
                or not (episode["terminated"] or episode["truncated"])):
            raise ValueError("invalid completed episode boundary")
    mean = sum(episode["episode_return"] for episode in episodes) / len(episodes) if episodes else None
    if mean is not None:
        require_numbers(mean)
    return dict(completed_episodes=len(episodes),
                natural_episodes=sum(ep["terminated"] and not ep["truncated"] for ep in episodes),
                truncated_episodes=sum(ep["truncated"] for ep in episodes),
                positive_return_natural_episodes=sum(ep["terminated"] and not ep["truncated"] and ep["episode_return"] > 0 for ep in episodes),
                mean_completed_return=mean)


def require_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            require_numbers(child)
    elif isinstance(value, list):
        for child in value:
            require_numbers(child)
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("learner report contains a non-finite or non-numeric value")


def audit(path):
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    with Path(path).open() as source:
        header = json.loads(next(source))
        check(header["event"] == "run_start" and header["protocol"] in ("kindle-vector-v1", VECTOR_PROTOCOL), "unknown vector protocol")
        count, config = header["num_envs"], header["config"]
        check(type(count) is int and count > 0, "invalid stream count")
        check(header["steps"] > 0 and header["steps"] % count == 0, "invalid action budget")
        check(len(header["environment_seeds"]) == count and len(set(header["environment_seeds"])) == count, "environment seeds must be independent")
        check(header["mode"] in ("train", "evaluate_sample", "evaluate_greedy"), "unknown action mode")
        for field in ("batch_size", "batch_length", "replay_context", "replay_capacity", "action_count"):
            check(type(config[field]) is int and config[field] > 0, f"invalid {field}")
        require_numbers(config["train_ratio"])
        check(config["train_ratio"] >= 0, "invalid train ratio")
        training = header["mode"] == "train" and config["train_ratio"] > 0
        required = config["replay_context"] + config["batch_length"]
        samples = config["batch_size"] * config["batch_length"]
        credit_per_action = f32(f32(config["train_ratio"]) / samples)
        replay = deque(range(count))
        replay_lengths = [1] * count
        actions = updates = 0
        credit = 0.0
        started = False
        pending_updates = 0
        pending_episodes = set()
        pending_resets = set()
        episode_lengths = [0] * count
        episode_returns = [0.0] * count
        total_rewards = [0.0] * count
        episode_counts = [0] * count
        last_frames = [0] * count
        last_flags = None
        final = None
        completed = []

        def push(stream):
            if len(replay) == config["replay_capacity"]:
                replay_lengths[replay.popleft()] -= 1
            replay.append(stream)
            replay_lengths[stream] += 1

        def settled():
            check(not pending_updates and not pending_episodes and not pending_resets, "incomplete vector round")

        for line in source:
            event = json.loads(line)
            check(final is None, "events after run_end")
            kind = event["event"]
            if kind == "transition":
                settled()
                actions += count
                check(event["run_step"] == actions and event["vector_tick"] == actions // count, "vector/action counter mismatch")
                for field in ("actions", "rewards", "stored_rewards", "terminated", "truncated", "executed_action_frames"):
                    check(len(event[field]) == count, f"wrong {field} batch length")
                for stream in range(count):
                    action = event["actions"][stream]
                    check(type(action) is int and 0 <= action < config["action_count"], "invalid executed action")
                    reward = event["rewards"][stream]
                    require_numbers(reward)
                    require_numbers(event["stored_rewards"][stream])
                    check(len(event["stored_rewards"][stream]) == 2, "stored reward must separate extrinsic and intrinsic")
                    check(event["stored_rewards"][stream][0] == reward, "stored extrinsic reward differs")
                    terminal, truncated = event["terminated"][stream], event["truncated"][stream]
                    check(type(terminal) is bool and type(truncated) is bool, "invalid boundary flags")
                    frames = event["executed_action_frames"][stream]
                    check(1 <= frames - last_frames[stream] <= header["action_repeat"], "invalid actual frame increment")
                    last_frames[stream] = frames
                    episode_lengths[stream] += 1
                    episode_returns[stream] += reward
                    total_rewards[stream] += reward
                    push(stream)
                    if started:
                        credit = f32(credit + credit_per_action)
                    if terminal or truncated:
                        pending_episodes.add(stream)
                        pending_resets.add(stream)
                last_flags = event["terminated"], event["truncated"]
                ready = sum(max(0, length - required + 1) for length in replay_lengths) >= samples
                if training and ready:
                    if not started:
                        started, credit = True, 1.0
                    pending_updates = int(credit)
            elif kind == "learner":
                check(event["run_step"] == actions and pending_updates > 0, "unexpected learner update")
                report = event["report"]
                require_numbers(report)
                updates += 1
                check(report["learner_step"] == header["starting_learner_step"] + updates, "learner counter mismatch")
                check(report["replay_len"] == len(replay), "learner replay length mismatch")
                pending_updates -= 1
                credit = f32(credit - 1.0)
            elif kind == "episode":
                stream = event["stream"]
                check(not pending_updates and stream in pending_episodes, "unexpected episode")
                check(event["run_step"] == actions and event["stream_step"] == actions // count, "episode action counter mismatch")
                check(event["episode"] == episode_counts[stream], "episode index mismatch")
                check(event["episode_length"] == episode_lengths[stream], "episode length mismatch")
                check(event["episode_return"] == episode_returns[stream], "episode reward mismatch")
                check((event["terminated"], event["truncated"]) == (last_flags[0][stream], last_flags[1][stream]), "episode flags mismatch")
                pending_episodes.remove(stream)
                episode_counts[stream] += 1
                episode_lengths[stream] = 0
                episode_returns[stream] = 0.0
                completed.append(event)
            elif kind == "reset":
                check(not pending_updates and not pending_episodes, "reset before final observation was accounted")
                ids = event["streams"]
                check(event["run_step"] == actions and len(ids) == len(set(ids)) and set(ids) == pending_resets, "wrong stream reset")
                for stream in ids:
                    push(stream)
                pending_resets.clear()
            elif kind in ("progress", "run_end"):
                settled()
                check(event["run_step"] == actions and event["vector_ticks"] == actions // count, "progress action counter mismatch")
                check(event["environment_step"] == header["starting_environment_step"] + actions, "global action counter mismatch")
                check(event["learner_step"] == header["starting_learner_step"] + updates, "progress learner counter mismatch")
                check(event["replay_len"] == len(replay), "progress replay length mismatch")
                check(event["training_debt"] == credit, "lost or invented training credit")
                for field, expected in (("executed_action_frames", last_frames), ("total_rewards", total_rewards), ("episode_counts", episode_counts), ("partial_returns", episode_returns), ("partial_lengths", episode_lengths)):
                    check(event[field] == expected, f"{field} ledger mismatch")
                require_numbers(event["stage_seconds"])
                check(event["elapsed_seconds"] > 0 and math.isfinite(event["elapsed_seconds"]), "invalid clock")
                elapsed = event["elapsed_seconds"]
                check(event["actions_per_second"] == actions / elapsed, "invalid action throughput")
                check(event["aggregate_simulated_wall_ratio"] == sum(last_frames) / 60 / elapsed, "invalid aggregate game clock")
                check(event["per_stream_simulated_wall_ratio"] == [frames / 60 / elapsed for frames in last_frames], "invalid per-stream game clocks")
                if kind == "run_end":
                    final = event
                    check(final["reason"] in ("budget_complete", "interrupted"), "unknown stop reason")
                    check(actions <= header["steps"] and (final["reason"] != "budget_complete" or actions == header["steps"]), "incomplete declared budget")
                    check(final["learner_updates"] == updates, "final counts mismatch")
            elif kind == "checkpoint":
                settled()
                check(event["run_step"] == actions and event["learner_step"] == header["starting_learner_step"] + updates, "checkpoint counters mismatch")
            else:
                raise ValueError(f"unknown event {kind}")
        check(final is not None, "missing run_end")
        scores = episode_summary(completed)
        if header["protocol"] == "kindle-vector-v1":
            expected = dict(completed_games=scores["completed_episodes"],
                            natural_wins=scores["positive_return_natural_episodes"],
                            mean_completed_return=scores["mean_completed_return"])
            forbidden = set(scores) - {"mean_completed_return"}
        else:
            expected = scores
            forbidden = {"completed_games", "natural_wins"}
        check(not forbidden.intersection(final), "mixed vector episode-summary versions")
        for field, value in expected.items():
            check(field in final, f"missing final score field: {field}")
            if value is not None:
                require_numbers(final[field])
            check(final[field] == value and (field == "mean_completed_return" or type(final[field]) is int), "final score mismatch")
        return dict(path=str(path), protocol=header["protocol"], actions=actions, updates=updates, num_envs=count,
                    **scores,
                    budget_complete=final["reason"] == "budget_complete", accounting_valid=True)


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        print(json.dumps(audit(path)))

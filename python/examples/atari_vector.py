"""Independent Atari streams, one native batched LeVJEPA/Dreamer learner.

--steps counts aggregate executed actions, not vector ticks or per-env actions.
CPU environment stepping is synchronous; GPU inference is batched. Terminal
observations are consumed before individually resetting completed environments.
"""

import argparse
import inspect
import json
import math
import signal
import sys
import time
from pathlib import Path

import ale_py
import gymnasium as gym

import kindle
from kindle._vector_audit import EPISODE_EVALUATION_PROTOCOL, VECTOR_PROTOCOL, episode_summary
import kindle._exploration as exploration_module
from kindle._exploration import EXPLORATION_KIND, EXPLORATION_PROTOCOL, PersistentExploration
from atari import (
    ATARI_ACTION_REPEAT, ATARI_PROTOCOLS, DreamerAtariPreprocessing,
    checkpoint_identity, sha256_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atari-protocol", choices=ATARI_PROTOCOLS, default="published")
    parser.add_argument("--model-size", default="12m")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--world-microbatch-size", type=int)
    parser.add_argument("--train-ratio", type=float, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.00004)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=20000)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--episodes-per-env", type=int,
                        help="frozen restore only: stop when every stream completes this many episodes; steps is a hard action cap")
    parser.add_argument("--exploration-probability", type=float, default=0.0)
    parser.add_argument("--exploration-hold", type=int, default=16)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.steps % args.num_envs:
        parser.error("steps must be a positive multiple of num-envs")
    if args.report_every <= 0 or args.checkpoint_every <= 0:
        parser.error("report/checkpoint intervals must be positive")
    if args.world_microbatch_size is not None and args.world_microbatch_size <= 0:
        parser.error("world-microbatch-size must be positive")
    if args.greedy and not args.evaluate:
        parser.error("greedy actions are only supported for frozen evaluation")
    if args.episodes_per_env is not None:
        if args.episodes_per_env <= 0:
            parser.error("episodes-per-env must be positive")
        if not args.evaluate or not args.restore or args.checkpoint:
            parser.error("episodes-per-env requires frozen restore without checkpoint writes")
    if not math.isfinite(args.exploration_probability) or not 0 <= args.exploration_probability <= 1:
        parser.error("exploration probability must be in [0, 1]")
    if args.exploration_hold <= 0:
        parser.error("exploration hold must be positive")
    if args.evaluate and args.exploration_probability:
        parser.error("frozen evaluation must not use exploration overrides")
    training_options = {"--model-size", "--batch-size", "--batch-length", "--world-microbatch-size", "--train-ratio", "--learning-rate", "--exploration-probability", "--exploration-hold"}
    if args.restore and any(arg.split("=", 1)[0] in training_options for arg in sys.argv[1:]):
        parser.error("training overrides require a fresh run; restore uses checkpoint config")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must fit an unsigned 32-bit integer")
    if args.output.exists() or (args.checkpoint and args.checkpoint.exists()):
        parser.error("output and checkpoint must be fresh paths")
    if args.exploration_probability and "action_overrides" not in inspect.signature(kindle.VectorAgent.act).parameters:
        parser.error("persistent exploration requires native vector action overrides")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = ATARI_PROTOCOLS[args.atari_protocol]
    gym.register_envs(ale_py)
    environments = []
    output = args.output.open("x")

    def emit(event):
        event["unix_time"] = time.time()
        print(json.dumps(event, separators=(",", ":"), allow_nan=False), file=output, flush=True)

    try:
        env_seeds = [(args.seed + stream * 1_000_003) % 2**32 for stream in range(args.num_envs)]
        initial = []
        for seed in env_seeds:
            env = DreamerAtariPreprocessing(gym.make(
                args.environment, frameskip=1, repeat_action_probability=0.0,
                full_action_space=protocol.full_action_space,
            ), noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames)
            environments.append(env)
            initial.append(env.reset(seed=seed)[0])
        actions = int(environments[0].action_space.n)
        config = kindle.default_config(actions, args.model_size)
        config.update(seed=args.seed, batch_size=args.batch_size, batch_length=args.batch_length,
                      world_backprop_length=args.batch_length,
                      world_microbatch_size=(args.batch_size if args.world_microbatch_size is None else args.world_microbatch_size),
                      train_ratio=args.train_ratio, learning_rate=args.learning_rate,
                      learning_rate_warmup=1000, agc=0.3)
        config["loss_scales"].update(reconstruction=0.0, future_prediction=0.25)
        exploration = (PersistentExploration(dict(kind=EXPLORATION_KIND,
            probability=args.exploration_probability, hold_actions=args.exploration_hold,
            seed=args.seed), args.num_envs, actions) if args.exploration_probability else None)
        construction = time.perf_counter()
        restored = checkpoint_identity(args.restore) if args.restore else None
        agent = (kindle.VectorAgent.restore(str(args.restore), args.encoder_checkpoint, args.num_envs)
                 if args.restore else kindle.VectorAgent(args.encoder_checkpoint, args.num_envs, config))
        if agent.config["action_count"] != actions:
            raise ValueError("checkpoint action vocabulary differs from environment")
        construction = time.perf_counter() - construction
        ids = list(range(args.num_envs))
        starting_actions, starting_updates = agent.environment_step, agent.learner_step
        started = time.perf_counter()
        agent.begin_episodes(ids, initial)
        exploration_header = (dict(exploration=exploration.config,
            exploration_sha256=sha256_file(exploration_module.__file__)) if exploration else {})
        vector_protocol = (EPISODE_EVALUATION_PROTOCOL if args.episodes_per_env else
                           EXPLORATION_PROTOCOL if exploration else VECTOR_PROTOCOL)
        emit(dict(event="run_start", protocol=vector_protocol, environment=args.environment,
                  num_envs=args.num_envs, steps=args.steps, seed=args.seed, environment_seeds=env_seeds,
                  **(dict(evaluation_episodes_per_stream=args.episodes_per_env) if args.episodes_per_env else {}),
                  policy_seed_rule="config.seed + stream (wrapping u64)",
                  atari_protocol=args.atari_protocol, action_repeat=ATARI_ACTION_REPEAT,
                  noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames,
                  full_action_space=protocol.full_action_space, sticky_actions=0.0,
                  action_meanings=list(environments[0].action_meanings), ale_py_version=ale_py.__version__,
                  mode=("evaluate_greedy" if args.greedy else "evaluate_sample") if args.evaluate else "train",
                  config=agent.config, model_provenance=agent.provenance, gpu_device=agent.gpu_device,
                  cpu_worker_threads=agent.cpu_worker_threads,
                  trainable_parameter_counts=agent.trainable_parameter_counts,
                  restored_checkpoint=restored, starting_environment_step=starting_actions,
                  starting_learner_step=starting_updates, agent_construction_seconds=construction,
                  native_extension_sha256=sha256_file(kindle._native.__file__),
                  runner_sha256=sha256_file(__file__), wrapper_sha256=sha256_file(Path(__file__).with_name("atari.py")),
                  **exploration_header))
        stop = False

        def request_stop(_signum, _frame):
            nonlocal stop
            stop = True

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        episode_returns = [0.0] * args.num_envs
        episode_lengths = [0] * args.num_envs
        episode_counts = [0] * args.num_envs
        total_rewards = [0.0] * args.num_envs
        completed = []
        timing = dict(act=0.0, environment=0.0, observe=0.0, learn=0.0, reset=0.0, checkpoint=0.0)
        last_report = last_checkpoint = 0
        run_actions = 0

        def save():
            before = time.perf_counter()
            agent.save_checkpoint(str(args.checkpoint))
            emit(dict(event="checkpoint", run_step=run_actions, learner_step=agent.learner_step,
                      identity=checkpoint_identity(args.checkpoint)))
            timing["checkpoint"] += time.perf_counter() - before

        def progress(event):
            elapsed = time.perf_counter() - started
            frames = [env.executed_action_frames for env in environments]
            exploration_counts = (dict(overridden_actions=exploration.overridden_actions)
                                  if exploration else {})
            return dict(event=event, run_step=run_actions, vector_ticks=run_actions // args.num_envs,
                        environment_step=agent.environment_step, learner_step=agent.learner_step,
                        elapsed_seconds=elapsed, actions_per_second=run_actions / elapsed,
                        executed_action_frames=frames,
                        reset_noop_frames=[env.reset_noop_frames for env in environments],
                        emulator_resets=[env.emulator_resets for env in environments],
                        aggregate_simulated_wall_ratio=sum(frames) / 60 / elapsed,
                        per_stream_simulated_wall_ratio=[n / 60 / elapsed for n in frames],
                        total_rewards=total_rewards, episode_counts=episode_counts,
                        partial_returns=episode_returns, partial_lengths=episode_lengths,
                        replay_len=agent.replay_len, training_debt=agent.training_debt,
                        stage_seconds=timing, **exploration_counts)

        try:
            for tick in range(1, args.steps // args.num_envs + 1):
                if stop:
                    break
                before = time.perf_counter()
                overrides = exploration.actions() if exploration else None
                selected = (agent.act(greedy=args.greedy, action_overrides=overrides)
                            if exploration else agent.act(greedy=args.greedy))
                if overrides is not None and any(forced is not None and selected[stream] != forced
                                                for stream, forced in enumerate(overrides)):
                    raise ValueError("native action override was not honored")
                timing["act"] += time.perf_counter() - before
                before = time.perf_counter()
                results = [env.step(action) for env, action in zip(environments, selected)]
                timing["environment"] += time.perf_counter() - before
                frames, rewards, terminated, truncated, _ = map(list, zip(*results))
                before = time.perf_counter()
                stored_rewards = agent.observe(ids, frames, rewards, terminated, truncated)
                timing["observe"] += time.perf_counter() - before
                run_actions += args.num_envs
                emit(dict(event="transition", vector_tick=tick, run_step=run_actions,
                          actions=selected, rewards=rewards, stored_rewards=stored_rewards,
                          terminated=terminated, truncated=truncated,
                          executed_action_frames=[env.executed_action_frames for env in environments],
                          **(dict(action_overrides=overrides) if exploration else {})))
                before = time.perf_counter()
                reports = [] if args.evaluate else agent.learn_scheduled()
                timing["learn"] += time.perf_counter() - before
                for report in reports:
                    emit(dict(event="learner", run_step=run_actions, report=report))
                resets = []
                for stream in ids:
                    episode_returns[stream] += float(rewards[stream])
                    total_rewards[stream] += float(rewards[stream])
                    episode_lengths[stream] += 1
                    if terminated[stream] or truncated[stream]:
                        result = dict(event="episode", stream=stream, run_step=run_actions,
                                      stream_step=tick, episode=episode_counts[stream],
                                      episode_return=episode_returns[stream], episode_length=episode_lengths[stream],
                                      terminated=bool(terminated[stream]), truncated=bool(truncated[stream]))
                        emit(result)
                        completed.append(result)
                        episode_counts[stream] += 1
                        episode_returns[stream] = 0.0
                        episode_lengths[stream] = 0
                        resets.append(stream)
                before = time.perf_counter()
                if resets:
                    reset_frames = [environments[stream].reset()[0] for stream in resets]
                    agent.begin_episodes(resets, reset_frames)
                    if exploration:
                        exploration.reset(resets)
                    emit(dict(event="reset", run_step=run_actions, streams=resets))
                timing["reset"] += time.perf_counter() - before
                if args.checkpoint and run_actions - last_checkpoint >= args.checkpoint_every:
                    save()
                    last_checkpoint = run_actions
                if run_actions - last_report >= args.report_every:
                    event = progress("progress")
                    emit(event)
                    print(f"{run_actions}/{args.steps} actions; {event['actions_per_second']:.2f} actions/s; "
                          f"{agent.learner_step - starting_updates} updates; debt={agent.training_debt:.3f}", flush=True)
                    last_report = run_actions
                if args.episodes_per_env and min(episode_counts) >= args.episodes_per_env:
                    break
            if args.checkpoint and run_actions != last_checkpoint:
                save()
            event = progress("run_end")
            reason = "budget_complete"
            if args.episodes_per_env:
                reason = "episode_budget_complete" if min(episode_counts) >= args.episodes_per_env else "action_cap_reached"
            event.update(reason="interrupted" if stop else reason,
                         **episode_summary(completed),
                         learner_updates=agent.learner_step - starting_updates)
            emit(event)
            print(json.dumps(event, indent=2), flush=True)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    finally:
        output.close()
        for env in environments:
            env.close()


if __name__ == "__main__":
    main()

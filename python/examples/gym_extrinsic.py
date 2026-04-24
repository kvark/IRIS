"""Minimal gym harness that trains kindle using ONLY the extrinsic-reward
primitive (no homeo, no intrinsic primitives). Diagnostic: validates
that kindle's policy-gradient machinery works on a standard RL env
where we KNOW the reward signal is learnable.

If kindle learns `CartPole-v1` via `--extrinsic-alpha 1.0`, its core RL
stack is functional and the Atari plateau is a sample-efficiency /
exploration issue specific to sparse-reward imbalanced-event envs.
If CartPole also plateaus, there's a deeper structural issue to find.

Usage:
    python python/examples/gym_extrinsic.py --env CartPole-v1 --steps 30000
    python python/examples/gym_extrinsic.py --env Pendulum-v1 --steps 30000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed.", file=sys.stderr)
        return 1
    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--history-len", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--advantage-clamp", type=float, default=2.0)
    parser.add_argument("--extrinsic-alpha", type=float, default=1.0)
    parser.add_argument("--reward-homeostatic", type=float, default=0.0)
    parser.add_argument("--reward-surprise", type=float, default=0.1)
    parser.add_argument("--reward-novelty", type=float, default=0.1)
    parser.add_argument("--reward-order", type=float, default=0.1)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--policy-adv-global-clip", type=float, default=0.0,
                        help="L2 norm clip on the batch-wide advantage vector "
                        "before the policy update — the policy-gradient analog "
                        "of global-grad-norm clipping. Try 1.0-5.0. 0 disables.")
    parser.add_argument("--policy-lr-adaptive-target", type=float, default=0.0,
                        help="Target |pi_loss| magnitude. When EMA(|pi_loss|) "
                        "exceeds it, per-step LR scales by target/EMA. Simple "
                        "TRPO-style damping. Try 0.5-2.0. 0 disables.")
    parser.add_argument("--policy-lr-adaptive-ema", type=float, default=0.05)
    parser.add_argument("--value-bootstrap", action="store_true",
                        help="Enable TD-bootstrap on the value-head target in "
                        "the n-step path: V_target = Σ γ^k r_{t+k} + γ^n·V(s_{t+n}). "
                        "Turns sparse external rewards into dense TD gradients "
                        "via the Bellman equation. Requires n_step ≥ 2.")
    parser.add_argument("--gae-lambda", type=float, default=0.0,
                        help="GAE λ for advantage estimation. 0 = disabled "
                        "(plain n-step advantage). 0.95 is the PPO default. "
                        "Decouples value target from advantage — prevents "
                        "value-kills-advantage collapse on dense-reward envs. "
                        "Implies value_bootstrap slot.")
    parser.add_argument("--value-loss-coef", type=float, default=1.0,
                        help="Value-head loss coefficient (PPO vf_coef). "
                        "1.0 = current behavior (value gradient dominates "
                        "on dense-reward envs). 0.5 is standard PPO. Try "
                        "0.1-0.5 to let policy gradient actually steer the "
                        "shared optimizer.")
    parser.add_argument("--policy-update-interval", type=int, default=1,
                        help="Update policy only every N env-steps, then do "
                        "N gradient steps on the accumulated rollout "
                        "(A2C/PPO-style). 1 = per-env-step (default). "
                        "Try n_step+1 (e.g. 9) to keep the rollout fully "
                        "on-policy with respect to the collector.")
    parser.add_argument("--advantage-normalize", action="store_true",
                        help="Zero-mean / unit-std the advantages per batch "
                        "before the policy update (standard PPO/A2C trick). "
                        "Strips the \"V lags reward\" bias — critical early "
                        "in training when V hasn't caught up and every "
                        "advantage is same-sign.")
    parser.add_argument("--use-ppo", action="store_true",
                        help="Use the PPO clipped-surrogate policy loss. "
                        "Requires policy_update_interval > 1 to exercise the "
                        "ratio. Incompatible with L1 options for now.")
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2,
                        help="PPO clip radius ε; ratio is clamped to "
                        "[1-ε, 1+ε]. Standard 0.2.")
    parser.add_argument("--async-envs", action="store_true")
    args = parser.parse_args()

    thunks = [lambda i=i: gym.make(args.env) for i in range(args.lanes)]
    if args.async_envs:
        envs = gym.vector.AsyncVectorEnv(thunks, shared_memory=True)
    else:
        envs = gym.vector.SyncVectorEnv(thunks)
    obs_batch, _ = envs.reset(seed=args.seed)
    obs_dim = obs_batch.shape[-1]
    act_space = envs.single_action_space
    is_discrete = isinstance(act_space, gym.spaces.Discrete)
    num_actions = int(act_space.n) if is_discrete else int(np.prod(act_space.shape))
    print(
        f"env={args.env} lanes={args.lanes} obs_dim={obs_dim} "
        f"{'discrete' if is_discrete else 'continuous'} actions={num_actions}"
    )

    agent = kindle.BatchAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        batch_size=args.lanes,
        env_ids=[1 + i for i in range(args.lanes)],
        seed=args.seed,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_len=args.history_len,
        n_step=args.n_step,
        gamma=args.gamma,
        advantage_clamp=args.advantage_clamp,
        entropy_beta=args.entropy_beta,
        reward_homeostatic=args.reward_homeostatic,
        reward_surprise=args.reward_surprise,
        reward_novelty=args.reward_novelty,
        reward_order=args.reward_order,
        extrinsic_reward_alpha=args.extrinsic_alpha,
        policy_adv_global_clip=args.policy_adv_global_clip if args.policy_adv_global_clip > 0 else None,
        policy_lr_adaptive_target=args.policy_lr_adaptive_target if args.policy_lr_adaptive_target > 0 else None,
        policy_lr_adaptive_ema=args.policy_lr_adaptive_ema,
        value_bootstrap=args.value_bootstrap,
        gae_lambda=args.gae_lambda if args.gae_lambda > 0 else None,
        value_loss_coef=args.value_loss_coef,
        policy_update_interval=args.policy_update_interval,
        advantage_normalize=args.advantage_normalize,
        use_ppo=args.use_ppo,
        ppo_clip_eps=args.ppo_clip_eps,
    )
    print("agent ready (MLP encoder)")

    ep_returns = [[] for _ in range(args.lanes)]
    cur_ret = np.zeros(args.lanes, dtype=np.float64)
    ep_count = 0
    t0 = time.time()

    for step in range(args.steps):
        # Normalize obs lists to list-of-lists for kindle.act()
        obs_lists = [obs_batch[i].astype(np.float32).tolist() for i in range(args.lanes)]
        actions = agent.act(obs_lists)
        if is_discrete:
            actions_np = np.array(actions, dtype=np.int64)
        else:
            # kindle returns discrete indices even for continuous; map to box center
            actions_np = np.array(actions, dtype=np.float32)
        next_obs, rewards, terms, truncs, _ = envs.step(actions_np)
        cur_ret += rewards
        dones = np.logical_or(terms, truncs)

        # Extrinsic reward — the only training signal.
        agent.set_extrinsic_reward(rewards.astype(np.float32, copy=False))
        next_lists = [next_obs[i].astype(np.float32).tolist() for i in range(args.lanes)]
        agent.observe(next_lists, [int(a) for a in actions], homeostatic=[[] for _ in range(args.lanes)])

        for i, done in enumerate(dones):
            if done:
                ep_returns[i].append(float(cur_ret[i]))
                cur_ret[i] = 0.0
                ep_count += 1
                agent.mark_boundary(i)
        obs_batch = next_obs

        if args.log_every and step > 0 and step % args.log_every == 0:
            diags = agent.diagnostics()
            d = diags[0]
            all_recent = [r for lane_rets in ep_returns for r in lane_rets[-5:]]
            avg_ret = sum(all_recent) / max(1, len(all_recent))
            elapsed = time.time() - t0
            sps = step * args.lanes / max(1e-3, elapsed)
            print(
                f"step={step:>6} eps={ep_count:>3} avg_ret={avg_ret:+7.1f} "
                f"| wm={float(d['loss_world_model']):.3f} "
                f"pi={float(d['loss_policy']):+6.2f} "
                f"ent={float(d['policy_entropy']):.2f} "
                f"r={float(d['reward_mean']):+5.2f} "
                f"| {sps:5.0f} env-steps/s"
            )

    envs.close()
    elapsed = time.time() - t0
    total_eps = sum(len(r) for r in ep_returns)
    if total_eps > 0:
        mean_ret = sum(r for lane in ep_returns for r in lane) / total_eps
    else:
        mean_ret = float("nan")
    print(f"\n--- {args.env} summary ---")
    print(f"total env-steps: {args.steps * args.lanes}")
    print(f"episodes: {total_eps}, mean return: {mean_ret:+.2f}")
    print(
        f"wall: {elapsed:.1f}s, throughput: "
        f"{args.steps * args.lanes / max(1e-3, elapsed):.0f} env-steps/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

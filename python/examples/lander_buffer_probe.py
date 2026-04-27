"""Buffer-level diagnostic for LunarLander training.

Trains a kindle agent on LunarLander-v3 for a short budget, then dumps
the rollout buffer and answers three questions:

1. Does the world model predict the next latent well? Compares stored
   `pred_error_i = ||WM(z_{i-1}, a_i) - z_i||` against a no-op baseline
   `||z_i - z_{i-1}||`. Ratio < 1 means WM is doing real work; ratio >= 1
   means it's worse than predicting unchanged.

2. Is reward temporally dilated correctly? For each completed episode in
   the buffer, computes the discounted return from each step and compares
   to the stored value baseline `V(s_{t-1})`. If V tracks the discounted
   return (high correlation, low MAE), the credit assignment is propagating
   late-episode signal back to earlier states — i.e. stable lander
   positions near landing should carry higher V than risky ones.

3. Per-episode breakdown shown for the best/worst episode in the buffer.

Run:
    python python/examples/lander_buffer_probe.py --steps 30000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed.", file=sys.stderr)
        return 1
    import kindle

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--rollout-length", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-loss-coef", type=float, default=0.1)
    p.add_argument("--wm-only", action="store_true",
                   help="Freeze the policy session (lr_policy=0) so WM "
                   "trains in isolation — random-action coverage, no "
                   "policy commits, V never updates. Tests whether WM "
                   "can learn dynamics under any conditions.")
    p.add_argument("--tag", default="run", help="Label printed at top "
                   "of summary so multi-run logs stay readable.")
    p.add_argument("--entropy-beta", type=float, default=0.01)
    p.add_argument("--bootstrap-value-clamp", type=float, default=200.0)
    p.add_argument("--value-clip-scale", type=float, default=400.0)
    p.add_argument("--lr-drop-on-solve", type=float, default=100.0)
    p.add_argument("--wm-residual", action="store_true",
                   help="Predict z' = z + delta(z, a) instead of z' "
                   "directly. Makes the no-op baseline the identity.")
    p.add_argument("--wm-aux-loss-coef", type=float, default=0.0,
                   help="In-policy WM auxiliary loss coefficient. >0 "
                   "adds a residual WM head sharing the policy encoder, "
                   "trained against stop_gradient(next_z). Forces the "
                   "policy encoder to be dynamics-aware. Try 0.1-1.0.")
    p.add_argument("--recon-loss-coef", type=float, default=0.0,
                   help="Reconstruction decoder anti-collapse loss "
                   "coefficient. >0 adds a decoder z→obs' + MSE loss "
                   "vs. stop_gradient(obs). Forces encoder to retain "
                   "enough info to invert. Try 0.1-1.0.")
    p.add_argument("--reward-pred-loss-coef", type=float, default=0.0,
                   help="Reward-from-z auxiliary loss coefficient. >0 "
                   "adds a head z→r̂ + MSE loss vs. per-row reward. "
                   "Forces z to encode reward-predictive features. "
                   "Try 0.01-0.1 (reward scale ~100 squared = 10000, "
                   "small coef gives loss in O(10).)")
    p.add_argument("--solve-auto-multiple", type=float, default=1.2)
    p.add_argument("--solve-windows", type=int, default=2)
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--dump-n", type=int, default=512,
                   help="How many recent transitions per lane to dump.")
    args = p.parse_args()

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(args.env) for _ in range(args.lanes)]
    )
    obs, _ = envs.reset(seed=args.seed)
    obs_dim = obs.shape[-1]
    num_actions = int(envs.single_action_space.n)
    print(f"env={args.env} lanes={args.lanes} obs_dim={obs_dim} actions={num_actions}")

    agent = kindle.BatchAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        batch_size=args.lanes,
        env_ids=[1 + i for i in range(args.lanes)],
        seed=args.seed,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_step=8,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_loss_coef=args.value_loss_coef,
        entropy_beta=args.entropy_beta,
        extrinsic_reward_alpha=1.0,
        reward_homeostatic=0.0,
        reward_surprise=0.0,
        reward_novelty=0.0,
        reward_order=0.0,
        end_to_end_encoder=True,
        rollout_length=args.rollout_length,
        bootstrap_value_clamp=args.bootstrap_value_clamp,
        value_clip_scale=args.value_clip_scale,
        advantage_normalize=True,
        wm_residual=args.wm_residual,
        wm_aux_loss_coef=args.wm_aux_loss_coef,
        recon_loss_coef=args.recon_loss_coef,
        reward_pred_loss_coef=args.reward_pred_loss_coef,
    )
    if args.wm_only:
        agent.set_lr_policy(0.0)
        print("wm-only: policy LR forced to 0; only WM + credit train")
    print(f"agent ready  [tag={args.tag} vlc={args.value_loss_coef} "
          f"wm_only={args.wm_only}]")

    ep_returns = [[] for _ in range(args.lanes)]
    cur_ret = np.zeros(args.lanes, dtype=np.float64)
    ep_count = 0
    lr_dropped = False
    auto_baseline = []
    solve_streak = 0
    t0 = time.time()

    for step in range(args.steps):
        obs_lists = [obs[i].astype(np.float32).tolist() for i in range(args.lanes)]
        actions = agent.act(obs_lists)
        actions_np = np.array(actions, dtype=np.int64)
        next_obs, rewards, terms, truncs, _ = envs.step(actions_np)
        cur_ret += rewards
        dones = np.logical_or(terms, truncs)

        agent.set_extrinsic_reward(rewards.astype(np.float32, copy=False))
        next_lists = [next_obs[i].astype(np.float32).tolist() for i in range(args.lanes)]
        agent.observe(next_lists, [int(a) for a in actions], homeostatic=[[] for _ in range(args.lanes)])

        for i, done in enumerate(dones):
            if done:
                ep_returns[i].append(float(cur_ret[i]))
                cur_ret[i] = 0.0
                ep_count += 1
                agent.mark_boundary(i)
        obs = next_obs

        if args.log_every and step > 0 and step % args.log_every == 0:
            d = agent.diagnostics()[0]
            recent = [r for lane in ep_returns for r in lane[-5:]]
            avg_ret = sum(recent) / max(1, len(recent))

            if not lr_dropped and args.lr_drop_on_solve > 0 and ep_count > 0:
                auto_baseline.append(avg_ret)
                if len(auto_baseline) > 2:
                    base = sum(auto_baseline[:2]) / 2
                    threshold = (args.solve_auto_multiple * base
                                 if base >= 0 else base / args.solve_auto_multiple)
                    if avg_ret >= threshold:
                        solve_streak += 1
                    else:
                        solve_streak = 0
                    if solve_streak >= args.solve_windows:
                        new_lr = args.lr / args.lr_drop_on_solve
                        agent.set_learning_rate(new_lr)
                        agent.set_lr_policy(new_lr / 2)
                        lr_dropped = True
                        print(f"[lr-drop] step={step} avg_ret={avg_ret:+.1f} "
                              f"threshold={threshold:+.1f} → lr {new_lr:.1e}")
            sps = step * args.lanes / max(1e-3, time.time() - t0)
            vs = np.array(agent.values(), dtype=np.float32)
            wm_aux_str = ""
            if args.wm_aux_loss_coef > 0:
                pred = float(agent.last_wm_aux_pred_mse())
                noop = float(agent.last_wm_aux_noop_mse())
                ratio = pred / max(1e-9, noop)
                wm_aux_str = (f" | aux pred={pred:.4f} noop={noop:.4f} "
                              f"r={ratio:.2f}")
            print(f"step={step:>6} eps={ep_count:>3} avg_ret={avg_ret:+7.1f} "
                  f"| wm={float(d['loss_world_model']):.3f} "
                  f"pi={float(d['loss_policy']):+5.2f} "
                  f"ent={float(d['policy_entropy']):.2f} "
                  f"V[{vs.min():+5.2f},{vs.max():+5.2f}]{wm_aux_str} "
                  f"| {sps:5.0f} sps")

    envs.close()
    print()
    print("=" * 78)
    print("BUFFER PROBE — WM prediction quality and reward temporal dilation")
    print("=" * 78)

    # Aggregate transitions across all lanes.
    all_buf = []
    for lane in range(args.lanes):
        all_buf.append(agent.dump_buffer(lane, args.dump_n))
    total_steps = sum(len(b) for b in all_buf)
    print(f"\ndumped {total_steps} transitions across {args.lanes} lanes "
          f"({args.dump_n} per lane requested).")

    # ---- Q1. WM prediction quality ----
    # pred_error stored per transition is ||WM(z_{i-1}, a_i) - z_i||.
    # No-op baseline: ||z_i - z_{i-1}||. Both skip env_boundary rows
    # (where z_{i-1} crossed an episode reset and the WM target wasn't
    # this transition's z).
    pred_errs = []
    noop_errs = []
    pred_err_squared_diff_to_noop_squared = []
    for buf in all_buf:
        for i in range(1, len(buf)):
            t_prev, t_cur = buf[i - 1], buf[i]
            if t_cur["env_boundary"]:
                continue
            z_prev = np.asarray(t_prev["latent"], dtype=np.float32)
            z_cur = np.asarray(t_cur["latent"], dtype=np.float32)
            noop = float(np.linalg.norm(z_cur - z_prev))
            pe = float(t_cur["pred_error"])
            pred_errs.append(pe)
            noop_errs.append(noop)
            pred_err_squared_diff_to_noop_squared.append((pe ** 2) - (noop ** 2))

    pred_errs = np.array(pred_errs)
    noop_errs = np.array(noop_errs)
    n_steps = len(pred_errs)
    if n_steps == 0:
        print("\n[WM] no usable transitions (every row was an env_boundary).")
    else:
        ratio = pred_errs.mean() / max(1e-9, noop_errs.mean())
        print(f"\n[WM]  N={n_steps}")
        print(f"      stored pred_error  ‖WM(z,a) − z'‖   "
              f"mean={pred_errs.mean():.4f}  median={np.median(pred_errs):.4f}  "
              f"p95={np.percentile(pred_errs, 95):.4f}")
        print(f"      no-op baseline     ‖z' − z‖         "
              f"mean={noop_errs.mean():.4f}  median={np.median(noop_errs):.4f}  "
              f"p95={np.percentile(noop_errs, 95):.4f}")
        print(f"      ratio (lower = WM working): {ratio:.3f}  "
              f"({'WM beats no-op' if ratio < 1 else 'WM WORSE than no-op — broken'})")

    # ---- Q2. Reward temporal dilation: V(s_{t-1}) vs Σ γ^k r_{t+k} ----
    # transition[i].value is V at the state PRIOR to action i (cached at
    # act() time). So compare it to the discounted return from step i
    # onward. Within a lane, walk forward until env_boundary or buffer end.
    gamma = args.gamma
    paired_v = []
    paired_ret = []
    per_episode_summaries = []  # list of (lane_idx, start_idx, length, total_r, mean_pe)
    for lane_idx, buf in enumerate(all_buf):
        # Identify episode boundaries: indices where transition[i] starts
        # a new episode (env_boundary=True) or i==0.
        ep_starts = [0] + [i for i in range(len(buf)) if buf[i]["env_boundary"]]
        # An episode runs from start S to next start S2−1 (or end).
        for k, s in enumerate(ep_starts):
            e = ep_starts[k + 1] if k + 1 < len(ep_starts) else len(buf)
            length = e - s
            if length < 4:
                continue
            rewards = np.asarray(
                [buf[j]["reward"] for j in range(s, e)], dtype=np.float64
            )
            # Discounted return from each step onward (inside this episode).
            disc = np.zeros(length, dtype=np.float64)
            running = 0.0
            for j in range(length - 1, -1, -1):
                running = rewards[j] + gamma * running
                disc[j] = running
            for j in range(length):
                paired_v.append(buf[s + j]["value"])
                paired_ret.append(disc[j])
            per_episode_summaries.append(
                (lane_idx, s, length, float(rewards.sum()),
                 float(np.mean([buf[j]["pred_error"] for j in range(s, e)])))
            )

    paired_v = np.array(paired_v, dtype=np.float64)
    paired_ret = np.array(paired_ret, dtype=np.float64)
    if len(paired_v) > 1:
        corr = float(np.corrcoef(paired_v, paired_ret)[0, 1])
        mae = float(np.mean(np.abs(paired_v - paired_ret)))
        bias = float(np.mean(paired_v - paired_ret))
        print(f"\n[V]   N={len(paired_v)} step-pairs across all in-buffer episodes")
        print(f"      V vs. discounted-return-from-this-step:")
        print(f"        corr  = {corr:+.3f}   "
              f"({'tracking' if corr > 0.3 else 'NOT tracking'})")
        print(f"        MAE   = {mae:.2f}")
        print(f"        bias V−R = {bias:+.2f}   "
              f"({'V over-estimates' if bias > 0 else 'V under-estimates'})")
        print(f"        V range  = [{paired_v.min():+.2f}, {paired_v.max():+.2f}]")
        print(f"        R range  = [{paired_ret.min():+.2f}, {paired_ret.max():+.2f}]")

    # ---- Q3. Per-episode breakdown ----
    if per_episode_summaries:
        per_episode_summaries.sort(key=lambda r: r[3])  # by total reward
        worst = per_episode_summaries[0]
        best = per_episode_summaries[-1]
        median = per_episode_summaries[len(per_episode_summaries) // 2]
        print(f"\n[EP]  in-buffer episodes: {len(per_episode_summaries)}")
        for label, ep in [("worst ", worst), ("median", median), ("best  ", best)]:
            print(f"      {label}: lane={ep[0]} len={ep[2]:>4}  "
                  f"total_r={ep[3]:+8.2f}  mean_pred_err={ep[4]:.4f}")

        # Step-by-step trace of the best episode.
        lane_idx, s, length, total_r, _ = best
        buf = all_buf[lane_idx]
        print(f"\n[BEST EPISODE TRACE]  lane={lane_idx}  total_r={total_r:+.1f}  "
              f"length={length}")
        print(f"      Lander obs[:6] meanings: x, y, vx, vy, angle, ang_vel")
        print(f"      step  obs[:6]                                "
              f"act  reward    V       discR    pred_err")
        # Sample evenly-spaced steps + the last 5.
        if length <= 12:
            shown_idx = list(range(length))
        else:
            shown_idx = list(np.linspace(0, length - 6, 6, dtype=int)) \
                        + list(range(length - 5, length))
        # Discounted-from-step return cache.
        rewards = np.asarray([buf[s + j]["reward"] for j in range(length)],
                             dtype=np.float64)
        disc = np.zeros(length, dtype=np.float64)
        running = 0.0
        for j in range(length - 1, -1, -1):
            running = rewards[j] + gamma * running
            disc[j] = running
        for j in shown_idx:
            t = buf[s + j]
            o = t["observation"][:6]
            obs_s = " ".join(f"{x:+5.2f}" for x in o)
            act = int(np.argmax(t["action"])) if t["action"] else -1
            print(f"      {j:>4}  {obs_s}  "
                  f"{act:>3}  {t['reward']:+7.2f}  "
                  f"{t['value']:+6.2f}  {disc[j]:+7.2f}  {t['pred_error']:.4f}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

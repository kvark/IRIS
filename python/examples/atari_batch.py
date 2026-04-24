"""Vectorized Atari training harness for kindle (BatchAgent + CNN encoder).

Target: hard focus on training throughput. Uses gymnasium vectorized envs
(SyncVectorEnv by default; AsyncVectorEnv when --async is set) and
measures where time goes: env.step, frame conversion, agent.act,
agent.observe. A baseline for later optimizations (external-buffer
plumbing, pure-GPU framecpy).

Standard Atari preprocessing:
- frameskip=4 inside ALE
- 84×84 grayscale
- framestack=4 → (C=4, H=84, W=84) per lane per step
- uint8 → float32 / 255 at hand-off to kindle

Usage:
    python python/examples/atari_batch.py --game Pong --lanes 8 --steps 5000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import numpy as np


# --- Ablation blocks ------------------------------------------------------
#
# Mirror arc_agi3_batch.py's design: named blocks compose via --preset /
# --enable / --disable. Explicit --flag overrides block defaults.

BLOCKS: dict[str, dict[str, object]] = {
    "rnd": {
        "rnd_alpha": 1.0,
    },
    "m8": {
        "delta_goal_alpha": 0.3,
        "delta_goal_threshold": 0.0,
        "delta_goal_merge_radius": 0.1,
        "delta_goal_surprise_threshold": 0.5,
    },
    "xeps": {
        "xeps_alpha": 0.3,
        "xeps_grid_resolution": 0.05,
    },
    "planner": {
        "planner_horizon": 4,
        "planner_samples": 32,
        "planner_every": 20,
    },
}

PRESETS: dict[str, list[str]] = {
    "none": [],
    "rnd_only": ["rnd"],
    "everything": ["rnd", "m8", "xeps", "planner"],
}


def _apply_blocks(args: argparse.Namespace, raw_argv: list[str]) -> list[str]:
    explicit: set[str] = set()
    for tok in raw_argv:
        if tok.startswith("--"):
            name = tok[2:].split("=", 1)[0]
            explicit.add(name.replace("-", "_"))

    active: list[str] = list(PRESETS.get(args.preset, []))
    if args.enable:
        for b in args.enable.split(","):
            b = b.strip()
            if b and b not in active:
                if b not in BLOCKS:
                    raise SystemExit(f"unknown block {b!r}; available: {sorted(BLOCKS)}")
                active.append(b)
    if args.disable:
        drop = {b.strip() for b in args.disable.split(",") if b.strip()}
        active = [b for b in active if b not in drop]

    for block_name in active:
        for arg_name, val in BLOCKS[block_name].items():
            if arg_name in explicit:
                continue
            setattr(args, arg_name, val)
    return sorted(active)


# Atari input shape after preprocessing.
N_CHANNELS = 4
FRAME_H = 84
FRAME_W = 84
FRAME_SIZE = N_CHANNELS * FRAME_H * FRAME_W  # 28224 floats per lane


def _make_env(game: str, seed: int):
    """Factory returning a fully-preprocessed Atari env callable for
    gymnasium's vector-env API."""
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    def _thunk():
        env = gym.make(f"ALE/{game}-v5", frameskip=1)
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            screen_size=FRAME_W,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,  # we normalize on the Rust/kindle side eventually
        )
        env = FrameStackObservation(env, stack_size=N_CHANNELS)
        env.reset(seed=seed)
        return env

    return _thunk


def _write_frames_into_shared(obs_batch: np.ndarray, dest: np.ndarray) -> None:
    """obs_batch: (num_envs, C, H, W) uint8 from gymnasium.
    dest: same-shape writable numpy view over kindle's shared host
    allocation (float32). Normalizes uint8→float32/255 and writes the
    result into `dest` in one pass — no intermediate allocation and
    no Python list conversion. The destination memory is the
    Vulkan-imported buffer; when this function returns, the WM
    session already "sees" the new frames.
    """
    if obs_batch.ndim == 5 and obs_batch.shape[-1] == 1:
        obs_batch = obs_batch[..., 0]
    # np.divide with an out= parameter performs the cast + scale in
    # a single loop and writes directly into the shared allocation.
    np.divide(obs_batch, 255.0, out=dest, dtype=np.float32, casting="unsafe")


def homeo_for(step_reward: float) -> list[dict[str, float]]:
    """Per-step env reward as a homeo variable.

    value = -step_reward, target = 0, tolerance = 0.5.

    Most Atari steps produce 0 reward (deviation 0, inside tolerance →
    homeo 0). A score event produces ±1, giving an immediate ±1 delta
    that's sharp in time — exactly the per-step signal the policy
    gradient needs. Cumulative-score formulations end up constant
    mid-episode and the value head learns the DC offset, zeroing the
    advantage; this delta formulation keeps the signal in the
    advantage, not the value."""
    return [
        {"value": -step_reward, "target": 0.0, "tolerance": 0.5},
    ]


class EventBalancer:
    """Rebalances per-step reward so rare events get amplified.

    Stock on-policy RL sees 21 losses per 1 win on Pong (random policy),
    so accumulated gradient tells the policy "avoid actions during
    losses" — which happens to be "don't chase the ball." This
    rebalancer tracks running counts of positive and negative events
    per lane and scales the rarer sign up so each event class
    contributes equally to the accumulated gradient.

    At balance (pos_count == neg_count) scale is 1 for both. In
    Pong's 21:1 regime, scale for positive events becomes ~21 and
    for negative stays 1 (or vice versa depending on which is rarer).

    Not pure task-agnostic — it embeds a prior that rare events are
    important — but it's a generic prior applicable to any sparse-
    reward env, not Atari-specific.
    """

    def __init__(self, num_lanes: int, eps: float = 1.0):
        import numpy as np
        self.pos = np.full(num_lanes, eps, dtype=np.float64)
        self.neg = np.full(num_lanes, eps, dtype=np.float64)

    def scale(self, rewards: np.ndarray) -> np.ndarray:
        """Return rebalanced rewards in-place-safe."""
        import numpy as np
        # Update counts first, so current-step scaling reflects the
        # distribution AFTER this step's contribution is known.
        self.pos += (rewards > 0).astype(np.float64)
        self.neg += (rewards < 0).astype(np.float64)
        # Scale factor amplifies the rarer sign to match the other.
        out = rewards.astype(np.float32).copy()
        pos_scale = self.neg / self.pos  # >1 when positives rare
        neg_scale = self.pos / self.neg  # >1 when negatives rare
        out[rewards > 0] *= pos_scale[rewards > 0].astype(np.float32)
        out[rewards < 0] *= neg_scale[rewards < 0].astype(np.float32)
        return out


class Profiler:
    """Tiny cumulative-time instrument with named sections."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def tick(self, section: str, elapsed: float) -> None:
        self.totals[section] = self.totals.get(section, 0.0) + elapsed
        self.counts[section] = self.counts.get(section, 0) + 1

    def report(self, total_elapsed: float) -> None:
        print("\n--- profile (cumulative wall time) ---")
        rows = sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
        for section, t in rows:
            pct = 100.0 * t / max(1e-9, total_elapsed)
            mean_ms = 1000.0 * t / max(1, self.counts[section])
            print(f"  {section:<18} {t:>7.2f}s  {pct:>5.1f}%  mean={mean_ms:>6.3f}ms")
        print(f"  {'TOTAL':<18} {total_elapsed:>7.2f}s")


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed. Try: pip install 'gymnasium[atari]'", file=sys.stderr)
        return 1

    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="none",
                        help="Named block combination. 'rnd_only' = RND curiosity "
                        "on top of bare kindle. 'everything' stacks all primitives "
                        "for collision testing. 'none' (default) runs bare kindle.")
    parser.add_argument("--enable", default=None,
                        help=f"Comma-separated blocks to add. Blocks: {','.join(sorted(BLOCKS.keys()))}.")
    parser.add_argument("--disable", default=None,
                        help="Comma-separated blocks to remove from the active set.")
    parser.add_argument("--game", default="Pong",
                        help="ALE game id (e.g. Pong, Breakout, Boxing, MsPacman).")
    parser.add_argument("--lanes", type=int, default=8,
                        help="Parallel env count (vectorized). Default 8.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Synchronous steps per lane.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--score-target", type=float, default=21.0,
                        help="Homeo target for episode cumulative score. Pong=21, "
                        "Breakout larger, etc. Used only for the homeo variable.")
    parser.add_argument("--async-envs", action="store_true",
                        help="Use gymnasium AsyncVectorEnv (multiprocessing). "
                        "Default is SyncVectorEnv (single process).")
    parser.add_argument("--reward-homeostatic", type=float, default=None,
                        help="Weight on the homeostatic primitive. Kindle default 2.0. "
                        "With per-step reward-delta homeo, try 0.5-1.0.")
    parser.add_argument("--reward-surprise", type=float, default=None)
    parser.add_argument("--reward-novelty", type=float, default=None)
    parser.add_argument("--reward-order", type=float, default=None)
    parser.add_argument("--advantage-clamp", type=float, default=None)
    parser.add_argument("--entropy-beta", type=float, default=None)
    parser.add_argument("--history-len", type=int, default=None,
                        help="L0 credit-attention window in steps. Default 16. "
                        "Atari score events land ~10-30 env-steps after the "
                        "causal action; try 64-128.")
    parser.add_argument("--n-step", type=int, default=None,
                        help="Advantage lookahead horizon. Default 1 (single-step). "
                        "N ≥ 2 trains on an N-step γ-discounted Monte-Carlo return — "
                        "needed when reward events are many steps downstream of the "
                        "causal action (Pong scoring, Breakout brick bounces).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor for n-step returns. Default 0.95. "
                        "Try 0.99 on long-horizon envs.")
    parser.add_argument("--value-bootstrap", action="store_true",
                        help="TD-bootstrap the value-head target "
                        "(V_target = Σ γ^k r_{t+k} + γ^n·V(s_{t+n})). "
                        "Densifies sparse-reward TD gradient.")
    parser.add_argument("--balance-events", action="store_true",
                        help="Harness-side rebalance per-step rewards so rare "
                        "events (positive in Pong/Breakout) get amplified to "
                        "contribute equally to the accumulated policy "
                        "gradient. Without it, kindle's policy commits in the "
                        "wrong direction on imbalanced envs (observed on Pong "
                        "at 400k env-steps: entropy drops 1.79→1.68 but "
                        "avg_return drifts −20.2→−21.0).")
    parser.add_argument("--extrinsic-alpha", type=float, default=None,
                        help="Weight on kindle's first-class extrinsic-reward "
                        "primitive. When > 0, the harness supplies the raw "
                        "env reward ± directly to kindle's per-step reward "
                        "channel (additive alongside surprise/novelty/homeo/…). "
                        "Unlike homeo's distance-to-target, this is signed and "
                        "passes through — the correct integration for "
                        "per-step ±1 Atari-style reward.")
    parser.add_argument("--rnd-alpha", type=float, default=None)
    parser.add_argument("--delta-goal-alpha", type=float, default=None)
    parser.add_argument("--delta-goal-threshold", type=float, default=None)
    parser.add_argument("--delta-goal-merge-radius", type=float, default=None)
    parser.add_argument("--delta-goal-surprise-threshold", type=float, default=None)
    parser.add_argument("--xeps-alpha", type=float, default=None)
    parser.add_argument("--xeps-grid-resolution", type=float, default=None)
    parser.add_argument("--planner-horizon", type=int, default=0)
    parser.add_argument("--planner-samples", type=int, default=None)
    parser.add_argument("--planner-every", type=int, default=0)
    args = parser.parse_args()
    active_blocks = _apply_blocks(args, sys.argv[1:])
    if active_blocks:
        print(f"active blocks: {','.join(active_blocks)}")
    else:
        print("active blocks: (none — bare kindle)")

    # --- Build vectorized envs ---
    thunks = [_make_env(args.game, args.seed + i) for i in range(args.lanes)]
    if args.async_envs:
        envs = gym.vector.AsyncVectorEnv(thunks, shared_memory=True)
    else:
        envs = gym.vector.SyncVectorEnv(thunks)
    obs_batch, _info = envs.reset(seed=args.seed)
    # gymnasium returns (num_envs, C, H, W, 1) when grayscale_newaxis=True or
    # (num_envs, C, H, W) when False. We configured False, so expect 4D.
    print(
        f"env={args.game} lanes={args.lanes} obs={obs_batch.shape} "
        f"dtype={obs_batch.dtype} action_space={envs.single_action_space}"
    )
    num_actions = int(envs.single_action_space.n)

    # --- Build kindle BatchAgent with CNN encoder ---
    # With CNN encoder, obs_dim is the SECONDARY obs-token width (fed to
    # reward primitives that need a flat token, e.g. order digest). It
    # must be ≤ OBS_TOKEN_DIM=64. The raw 84×84×4 frame travels via
    # set_visual_obs(), not the obs token.
    agent_kwargs: dict[str, Any] = dict(
        obs_dim=64,
        num_actions=num_actions,
        batch_size=args.lanes,
        env_ids=[1 + i for i in range(args.lanes)],
        seed=args.seed,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_kind="cnn",
        encoder_channels=N_CHANNELS,
        encoder_height=FRAME_H,
        encoder_width=FRAME_W,
    )
    for name in [
        "rnd_alpha", "delta_goal_alpha", "delta_goal_threshold",
        "delta_goal_merge_radius", "delta_goal_surprise_threshold",
        "xeps_alpha", "xeps_grid_resolution", "planner_samples",
        "reward_homeostatic", "reward_surprise", "reward_novelty", "reward_order",
        "advantage_clamp", "entropy_beta",
        "history_len", "n_step", "gamma",
        "extrinsic_alpha",
    ]:
        v = getattr(args, name, None)
        if v is not None:
            agent_kwargs[
                {
                    "rnd_alpha": "rnd_reward_alpha",
                    "xeps_alpha": "xeps_reward_alpha",
                    "extrinsic_alpha": "extrinsic_reward_alpha",
                }.get(
                    name, name
                )
            ] = v
    if args.planner_horizon > 0:
        agent_kwargs["planner_horizon"] = args.planner_horizon
    if args.value_bootstrap:
        agent_kwargs["value_bootstrap"] = True

    agent = kindle.BatchAgent(**agent_kwargs)

    # --- Frame delivery path (pure device-local) ---
    # Meganeura allocates every graph input as Memory::Shared —
    # DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT. Writing into the
    # memoryview lands bytes directly in the GPU-side buffer, no
    # staging, no explicit upload, no external-memory import. One
    # memcpy per frame is the theoretical minimum for CPU-produced
    # observations.
    frame_mv = agent.visual_obs_memoryview()
    frame_buf = np.frombuffer(frame_mv, dtype=np.float32).reshape(
        args.lanes, N_CHANNELS, FRAME_H, FRAME_W
    )
    assert frame_buf.flags.writeable, "frame_buf must be writable"
    print(
        f"agent ready (CNN {N_CHANNELS}×{FRAME_H}×{FRAME_W} → {args.latent_dim}); "
        f"visual_obs buffer: {agent.visual_obs_host_size() / (1 << 10):.1f} KiB "
        f"(Memory::Shared, device-local + host-visible)"
    )

    balancer: EventBalancer | None = (
        EventBalancer(args.lanes) if args.balance_events else None
    )

    # --- Training loop ---
    cur_scores = np.zeros(args.lanes, dtype=np.float64)
    ep_returns: list[list[float]] = [[] for _ in range(args.lanes)]
    ep_count = 0
    prof = Profiler()
    # Pre-allocate obs token list (all-zeros) reused each step —
    # CNN mode ignores its content but the observe() API takes a
    # token sequence anyway.
    obs_token_small: list[list[float]] = [[0.0] * 64 for _ in range(args.lanes)]
    t0 = time.time()

    def push_frame(obs_batch_u8: np.ndarray) -> None:
        """Normalize uint8 NCHW → float32/255 directly into the
        device-local, host-visible `visual_obs` buffer. After this
        call, the WM session is ready to consume the new frame on
        its next step — no further agent API call needed."""
        if obs_batch_u8.ndim == 5 and obs_batch_u8.shape[-1] == 1:
            obs_batch_u8 = obs_batch_u8[..., 0]
        np.divide(
            obs_batch_u8, 255.0, out=frame_buf, dtype=np.float32, casting="unsafe"
        )

    for step in range(args.steps):
        # 1) Hand the current frame to kindle (one path-dependent copy).
        t = time.time()
        push_frame(obs_batch)
        prof.tick("push_frame", time.time() - t)

        # 2) Sample actions (batched).
        t = time.time()
        actions = agent.act(obs_token_small)
        prof.tick("agent.act", time.time() - t)

        # 3) Step envs.
        t = time.time()
        actions_np = np.array(actions, dtype=np.int64)
        obs_batch, rewards, terms, truncs, _info = envs.step(actions_np)
        prof.tick("envs.step", time.time() - t)

        cur_scores += rewards
        dones = np.logical_or(terms, truncs)

        # 4) Next-frame for observe()'s WM pass.
        t = time.time()
        push_frame(obs_batch)
        prof.tick("push_frame (next)", time.time() - t)

        t = time.time()
        shaped = balancer.scale(rewards) if balancer is not None else rewards
        homeos = [homeo_for(float(shaped[i])) for i in range(args.lanes)]
        if args.extrinsic_alpha is not None and args.extrinsic_alpha > 0:
            # Raw env reward, signed, passed through kindle's extrinsic
            # primitive. Shaped=balanced copy if --balance-events is on.
            agent.set_extrinsic_reward(shaped.astype(np.float32, copy=False))
        agent.observe(obs_token_small, [int(a) for a in actions], homeostatic=homeos)
        prof.tick("agent.observe", time.time() - t)

        # Episode bookkeeping.
        for i, done in enumerate(dones):
            if done:
                ep_returns[i].append(float(cur_scores[i]))
                cur_scores[i] = 0.0
                ep_count += 1
                agent.mark_boundary(i)

        if args.log_every and step > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step * args.lanes / max(1e-3, elapsed)
            all_recent = [r for lane_rets in ep_returns for r in lane_rets[-3:]]
            avg_ret = sum(all_recent) / max(1, len(all_recent))

            diags = agent.diagnostics()
            d = diags[0]

            def _safe(x, default=float("nan")):
                return float(x) if x is not None else default

            wm = _safe(d.get("loss_world_model"))
            pi = _safe(d.get("loss_policy"))
            ent = _safe(d.get("policy_entropy"))
            surp = _safe(d.get("reward_surprise"))
            nov = _safe(d.get("reward_novelty"))
            hom = _safe(d.get("reward_homeo"))
            rew = _safe(d.get("reward_mean"))
            print(
                f"step={step:>6} eps={ep_count:>3} avg_ret={avg_ret:+6.1f} "
                f"| wm={wm:.3f} pi={pi:+7.2f} ent={ent:.2f} "
                f"r={rew:+5.2f} surp={surp:+4.2f} nov={nov:+4.2f} hom={hom:+6.2f} "
                f"| {sps:5.0f} env-steps/s"
            )

    envs.close()

    # --- Summary ---
    elapsed = time.time() - t0
    sps = args.steps * args.lanes / max(1e-3, elapsed)
    total_soft = sum(len(r) for r in ep_returns)
    mean_ret = sum(r for lane in ep_returns for r in lane) / max(1, total_soft)
    print(f"\n--- {args.game} summary ---")
    print(f"total env-steps: {args.steps * args.lanes}")
    print(f"episodes: {total_soft}, mean return: {mean_ret:+.2f}")
    print(f"wall: {elapsed:.1f}s, throughput: {sps:.0f} env-steps/s")
    prof.report(elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

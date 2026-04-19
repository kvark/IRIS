"""Batched LunarLander exploration with a Phase-E multi-lane kindle agent.

Spins up N concurrent LunarLander envs, feeds them through a single
`kindle.BatchAgent` with `batch_size=N`, and prints per-lane learning
diagnostics alongside rolling-mean extrinsic return. Extrinsic reward
is for monitoring only — training is purely intrinsic (homeostatic +
surprise + novelty + order).

Usage:
    python python/examples/lunar_lander_batch.py --lanes 4 --steps 2000
"""

from __future__ import annotations

import argparse
import math
import sys
import time


# Fuel cost per gymnasium action:
#   0 = noop, 1 = left thruster, 2 = main engine, 3 = right thruster
# Values mirror the rough gymnasium internal shaping: main engine burns
# ~10× more propellant than the side thrusters.
_FUEL_COST = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}


def lunar_lander_homeo_v1(obs, action):
    """Original shaping (pre-2026-04-18). Dominated by a `-5` constant
    "not safely landed" offset that the value baseline mostly absorbs,
    leaving the crash-risk / tilt / speed gradients weak in mid-flight.
    Kept for A/B comparison.
    """
    altitude = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    angle = float(obs[4])
    speed = math.sqrt(vx * vx + vy * vy)
    leg1 = float(obs[6])
    leg2 = float(obs[7])
    proximity = math.exp(-max(0.0, altitude) * 2.0)
    descent = max(0.0, -vy)
    crash_risk = descent * proximity
    fuel = _FUEL_COST.get(int(action), 0.0)
    both_legs = 1.0 if (leg1 > 0 and leg2 > 0) else 0.0
    stopped = 1.0 if speed < 0.1 else 0.0
    not_safely_landed = 1.0 - both_legs * stopped
    return [
        {"value": crash_risk * 10.0, "target": 0.0, "tolerance": 0.0},
        {"value": abs(angle) * proximity, "target": 0.0, "tolerance": 0.05},
        {"value": speed * proximity, "target": 0.0, "tolerance": 0.1},
        {"value": fuel, "target": 0.0, "tolerance": 0.0},
        {"value": not_safely_landed * 5.0, "target": 0.0, "tolerance": 0.0},
    ]


def lunar_lander_homeo_v2(obs, action):
    """v2 shaping: continuous gradient toward the landed state.

    Changes from v1:
      - Drop the `not_safely_landed * 5.0` flat offset (it was a
        near-constant -5 baseline in mid-flight that the value head
        absorbed, killing advantage variance).
      - Add smooth altitude shaping — each unit of altitude above the
        tolerance contributes a penalty, so the agent gets a continuous
        downward-gradient signal regardless of position over the pad.
      - Add a smooth descent-rate target centered at a gentle negative
        velocity: reward slow controlled descent, penalize both fall
        and hover.
      - Keep the crash-risk and touchdown penalties unchanged — those
        are already local gradients, not offsets.
    """
    altitude = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    angle = float(obs[4])
    speed = math.sqrt(vx * vx + vy * vy)
    proximity = math.exp(-max(0.0, altitude) * 2.0)
    descent = max(0.0, -vy)
    crash_risk = descent * proximity
    fuel = _FUEL_COST.get(int(action), 0.0)
    return [
        {"value": crash_risk * 10.0, "target": 0.0, "tolerance": 0.0},
        {"value": abs(angle) * proximity, "target": 0.0, "tolerance": 0.05},
        {"value": speed * proximity, "target": 0.0, "tolerance": 0.1},
        {"value": fuel, "target": 0.0, "tolerance": 0.0},
        # Smooth altitude shaping: positive altitude beyond tolerance
        # is a continuous penalty. This is the gradient that tells the
        # policy to descend when it's far from the ground, complementing
        # crash_risk (which only fires *near* the ground).
        {"value": max(0.0, altitude), "target": 0.0, "tolerance": 0.1},
        # Target a gentle controlled descent — vy between -0.5 and 0
        # is "good." Above (ascent) or below (falling) are both penalty.
        {"value": vy, "target": -0.25, "tolerance": 0.25},
    ]


def lunar_lander_homeo_v3(obs, action):
    """v3 shaping: v2 + a softer landed-bonus.

    Keeps v2's altitude and descent-rate gradients but reintroduces a
    `not_safely_landed * 1.0` term (vs v1's ×5.0). The idea: give the
    policy a contrast at the actual landing event without swamping
    mid-flight gradients with a constant offset.
    """
    altitude = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    angle = float(obs[4])
    speed = math.sqrt(vx * vx + vy * vy)
    leg1 = float(obs[6])
    leg2 = float(obs[7])
    proximity = math.exp(-max(0.0, altitude) * 2.0)
    descent = max(0.0, -vy)
    crash_risk = descent * proximity
    fuel = _FUEL_COST.get(int(action), 0.0)
    both_legs = 1.0 if (leg1 > 0 and leg2 > 0) else 0.0
    stopped = 1.0 if speed < 0.1 else 0.0
    not_safely_landed = 1.0 - both_legs * stopped
    return [
        {"value": crash_risk * 10.0, "target": 0.0, "tolerance": 0.0},
        {"value": abs(angle) * proximity, "target": 0.0, "tolerance": 0.05},
        {"value": speed * proximity, "target": 0.0, "tolerance": 0.1},
        {"value": fuel, "target": 0.0, "tolerance": 0.0},
        {"value": max(0.0, altitude), "target": 0.0, "tolerance": 0.1},
        {"value": vy, "target": -0.25, "tolerance": 0.25},
        {"value": not_safely_landed * 1.0, "target": 0.0, "tolerance": 0.0},
    ]


def lunar_lander_homeo_v4(obs, action):
    """v4 shaping: v3 plus an explicit positive pull toward leg contact.

    Every v1–v3 homeo term is either zero or negative, so the policy
    only sees "don't do X" signals and has no asymmetric pull toward
    the landed state. v4 adds a leg-contact target: the homeo
    variable is `-(leg1 + leg2)` (so 0 in air, negative on contact)
    with target `-2.0` (both legs down) — contact reduces homeo
    deviation and produces a positive advantage for actions that
    end up with legs on the ground.
    """
    entries = lunar_lander_homeo_v3(obs, action)
    leg1 = float(obs[6])
    leg2 = float(obs[7])
    entries.append(
        {
            "value": -(leg1 + leg2),
            "target": -2.0,
            "tolerance": 0.1,
        }
    )
    return entries


# Backward-compat alias: old scripts/tests still call lunar_lander_homeo.
def lunar_lander_homeo(obs, action):
    return lunar_lander_homeo_v1(obs, action)


def lunar_lander_homeo_v5(obs, action):
    """v5 shaping: v3 with the `not_safely_landed` term amplified 50×.

    v3's `not_safely_landed * 1.0` (at homeo weight 2.0 → effective
    ±2 reward) is a +1-reward blip at the single terminal step,
    rounding-error against the cumulative −400 to −1000 homeo over
    a 90-step episode. The M6 mechanism check (2026-04-19) showed
    that this makes R̂ trained on episode-sum returns unable to
    distinguish soft from crash.

    v5 boosts the multiplier to 50 so a soft landing produces a
    ~+50 terminal delta relative to a crash — visible against the
    cumulative episode integral. A direct experimental test of
    "does M6 work once the reward target carries the signal?".
    """
    altitude = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    angle = float(obs[4])
    speed = math.sqrt(vx * vx + vy * vy)
    leg1 = float(obs[6])
    leg2 = float(obs[7])
    proximity = math.exp(-max(0.0, altitude) * 2.0)
    descent = max(0.0, -vy)
    crash_risk = descent * proximity
    fuel = _FUEL_COST.get(int(action), 0.0)
    both_legs = 1.0 if (leg1 > 0 and leg2 > 0) else 0.0
    stopped = 1.0 if speed < 0.1 else 0.0
    not_safely_landed = 1.0 - both_legs * stopped
    return [
        {"value": crash_risk * 10.0, "target": 0.0, "tolerance": 0.0},
        {"value": abs(angle) * proximity, "target": 0.0, "tolerance": 0.05},
        {"value": speed * proximity, "target": 0.0, "tolerance": 0.1},
        {"value": fuel, "target": 0.0, "tolerance": 0.0},
        {"value": max(0.0, altitude), "target": 0.0, "tolerance": 0.1},
        {"value": vy, "target": -0.25, "tolerance": 0.25},
        # 50× the v3 multiplier — the amplified terminal signal that
        # makes episode-sum returns differentiable between soft/crash.
        {"value": not_safely_landed * 50.0, "target": 0.0, "tolerance": 0.0},
    ]


_SHAPING_VARIANTS = {
    "v1": lunar_lander_homeo_v1,
    "v2": lunar_lander_homeo_v2,
    "v3": lunar_lander_homeo_v3,
    "v4": lunar_lander_homeo_v4,
    "v5": lunar_lander_homeo_v5,
}


# Per-component LunarLander obs scales (x, y, vx, vy, angle, ang_vel, leg1, leg2).
# Divide by these to keep every input roughly in [-2, 2] so the world-model
# optimizer doesn't see gradients dominated by velocity magnitudes.
_LL_SCALE = [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0]


def normalize_obs(obs):
    return [float(x) / s for x, s in zip(obs, _LL_SCALE)]


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed. Try: pip install 'gymnasium[box2d]'", file=sys.stderr)
        return 1

    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--lanes", type=int, default=4, help="concurrent LunarLander envs (N)")
    parser.add_argument("--steps", type=int, default=2_000, help="synchronous steps (per lane)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="world-model learning rate (default 1e-4 — LunarLander is harsh)")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="hold each sampled action for K env steps per lane (Phase E.v3 "
        "action-persistence; K=1 is classic reactive L0, K>1 stretches the "
        "effective credit horizon by K× with no graph change)",
    )
    parser.add_argument(
        "--lr-policy",
        type=float,
        default=None,
        help="policy+value head LR override. Default is lr × 0.5. Set higher "
        "if the policy's softmax stays stuck at uniform despite WM training.",
    )
    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=None,
        help="entropy regularization coefficient. Positive values encourage "
        "exploration (prevent softmax collapse); negative values force "
        "commitment. 0 disables. Applies only to discrete-action envs.",
    )
    parser.add_argument("--num-options", type=int, default=None,
                        help="L1 option count. 1 (default) = L0-only. ≥ 2 activates Phase G.")
    parser.add_argument("--option-horizon", type=int, default=None,
                        help="env steps per option (Phase G). Default 10.")
    parser.add_argument("--history-len", type=int, default=None,
                        help="L0 causal-attention credit window. Default 16. Widen to "
                        "attribute long-horizon reward (e.g. 64 for LunarLander descent).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor for n-step returns. Default 0.95. "
                        "Only consulted when --n-step >= 2.")
    parser.add_argument("--n-step", type=int, default=None,
                        help="Lookahead horizon for policy advantage. 1 (default) = "
                        "single-step (r_t − V). N >= 2 trains on a transition from "
                        "N steps ago with a γ-discounted Monte-Carlo return.")
    parser.add_argument("--outcome-alpha", type=float, default=None,
                        help="M6 learnable-reward bonus weight α. 0 (default) disables "
                        "the outcome-value head. ~0.1 is a sane starting point.")
    parser.add_argument("--outcome-lr", type=float, default=None,
                        help="LR for the outcome-value head. Defaults to learning_rate × 0.3.")
    parser.add_argument("--outcome-ep-len", type=int, default=None,
                        help="Per-episode trajectory cap for the outcome head. Default 256.")
    parser.add_argument("--outcome-clamp", type=float, default=None,
                        help="Symmetric cap on R̂ before the α multiply. Default 5.0. "
                        "Raise alongside α when probing whether M6 is correct-but-quiet.")
    parser.add_argument("--outcome-target",
                        choices=["episode_sum", "terminal_reward", "reward_to_go"],
                        default=None,
                        help="What the outcome head trains on. 'episode_sum' (default) = "
                        "centered Σ r_base over the episode (shared-per-episode target, M6 v1). "
                        "'terminal_reward' = last-step r_base. 'reward_to_go' = per-step "
                        "target Σ_{k>=t} r_k, gives differentiated intra-episode supervision.")
    parser.add_argument("--outcome-bonus", choices=["raw", "potential_delta"],
                        default=None,
                        help="How R̂ enters the reward. 'raw' (default) = α · R̂. "
                        "'potential_delta' = α · (R̂_t − R̂_{t-1}), the Ng-et-al shaping that "
                        "converts a state-value estimate into a per-step signal.")
    parser.add_argument("--outcome-window", type=int, default=None,
                        help="M6 v2: window size for the outcome head. 1 (default) = "
                        "single-frame R̂(z_t) — M6 v1. k >= 2 concatenates the last k "
                        "latents [z_{t-k+1}, ..., z_t] so the head sees trajectory "
                        "momentum, not just present state.")
    parser.add_argument(
        "--shaping",
        choices=list(_SHAPING_VARIANTS.keys()),
        default="v1",
        help="homeostatic shaping variant. v1 = original (legacy), "
        "v2 = with altitude + descent-rate shaping, no -5 constant offset.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="label-smoothing ε on the action target (0=none, 0.1=standard). "
        "Prevents the softmax from fully collapsing to deterministic by "
        "keeping gradient alive on every logit even at a one-hot softmax.",
    )
    args = parser.parse_args()

    shaping_fn = _SHAPING_VARIANTS[args.shaping]

    # One env per lane, each with a distinct seed so the lanes diverge.
    envs = [gym.make(args.env) for _ in range(args.lanes)]
    raw_obs_per_lane: list[list[float]] = []
    obs_lists: list[list[float]] = []
    for i, env in enumerate(envs):
        obs, _info = env.reset(seed=args.seed + i)
        raw = [float(x) for x in obs]
        raw_obs_per_lane.append(raw)
        obs_lists.append(normalize_obs(raw))
    obs_dim = len(obs_lists[0])
    num_actions = int(envs[0].action_space.n)

    print(
        f"env={args.env} lanes={args.lanes} obs_dim={obs_dim} num_actions={num_actions} "
        f"steps={args.steps} seed={args.seed} lr={args.lr} "
        f"action_repeat={args.action_repeat}"
    )

    # Distinct env_ids per lane so the agent builds per-lane task embeddings
    # (still shares the encoder — same universal obs/action tokens).
    env_ids = [1 + i for i in range(args.lanes)]
    agent = kindle.BatchAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        batch_size=args.lanes,
        env_ids=env_ids,
        seed=args.seed,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        action_repeat=args.action_repeat,
        lr_policy=args.lr_policy,
        entropy_beta=args.entropy_beta,
        label_smoothing=args.label_smoothing,
        num_options=args.num_options,
        option_horizon=args.option_horizon,
        history_len=args.history_len,
        gamma=args.gamma,
        n_step=args.n_step,
        outcome_reward_alpha=args.outcome_alpha,
        lr_outcome=args.outcome_lr,
        outcome_clamp=args.outcome_clamp,
        outcome_target=args.outcome_target,
        outcome_bonus=args.outcome_bonus,
        outcome_window=args.outcome_window,
        outcome_max_episode_len=args.outcome_ep_len,
    )
    print("agent ready (compiled graphs once, N lanes)")

    ep_returns: list[list[float]] = [[] for _ in range(args.lanes)]
    cur_returns: list[float] = [0.0 for _ in range(args.lanes)]
    # Per-episode outcome tagged from the final transition:
    #   "soft"  — both legs touching and low speed at termination (safe
    #             landing, anywhere on the terrain)
    #   "crash" — terminated with high speed or no legs down
    #   "timeout" — truncated without landing
    ep_outcomes: list[list[str]] = [[] for _ in range(args.lanes)]
    # Rolling window of outcome tags for "is the agent improving?"
    # Detection: we want per-window soft-rate, not cumulative, since
    # cumulative gets averaged down by the poor early episodes.
    from collections import deque
    recent_outcomes: deque[str] = deque(maxlen=200)
    window_log: list[tuple[int, int, int]] = []  # (step, soft_count, window_size)
    total_episodes = 0
    t0 = time.time()

    # M6 mechanism instrumentation: per-step r_hat per lane, tagged
    # with the episode's eventual outcome once it completes. Enables
    # the post-run discrimination check that verifies R̂ actually
    # encodes trajectory quality rather than some orthogonal confound.
    #   per_lane_cur_rhats[i] : list of r_hat over the current episode
    #   outcome_rhats[outcome] : list of (ep_len, mean, early_mean,
    #                              mid_mean, late_mean) tuples
    per_lane_cur_rhats: list[list[float]] = [[] for _ in range(args.lanes)]
    outcome_rhats: dict[str, list[tuple[int, float, float, float, float]]] = {
        "soft": [], "crash": [], "timeout": [],
    }

    for step in range(args.steps):
        actions = agent.act(obs_lists)

        # Step each env with its own action.
        raw_next: list[list[float]] = []
        next_lists: list[list[float]] = []
        dones: list[bool] = []
        terms: list[bool] = []
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
            raw = [float(x) for x in next_obs]
            cur_returns[i] += float(reward)
            done = bool(terminated or truncated)
            raw_next.append(raw)
            next_lists.append(normalize_obs(raw))
            dones.append(done)
            terms.append(bool(terminated))

        # Homeostatic signals use raw-space interpretations (altitude, vy, ...).
        # Pass the action so the fuel-cost term can charge the right amount.
        homeos = [
            shaping_fn(raw_next[i], actions[i]) for i in range(args.lanes)
        ]
        agent.observe(next_lists, actions, homeostatic=homeos)

        # M6 instrumentation: record per-step r_hat per lane (only
        # meaningful when M6 is active; we still accumulate but the
        # post-run summary is trivial if everything is zero).
        step_rhats = agent.r_hats()
        for i in range(args.lanes):
            per_lane_cur_rhats[i].append(float(step_rhats[i]))

        # Per-lane episode reset + boundary marking.
        for i, done in enumerate(dones):
            if done:
                # Classify terminal state from the final observation.
                leg1, leg2 = raw_next[i][6], raw_next[i][7]
                vx, vy = raw_next[i][2], raw_next[i][3]
                term_speed = math.sqrt(vx * vx + vy * vy)
                if terms[i] and leg1 > 0 and leg2 > 0 and term_speed < 0.25:
                    outcome = "soft"
                elif terms[i]:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                ep_outcomes[i].append(outcome)
                ep_returns[i].append(cur_returns[i])
                recent_outcomes.append(outcome)
                total_episodes += 1
                cur_returns[i] = 0.0
                # Snapshot this lane's per-step r_hat history and tag it
                # with the outcome. Compute three-phase means so we can
                # see whether R̂ discriminates early (while the state is
                # still far from terminal), late, or throughout.
                rhats = per_lane_cur_rhats[i]
                ep_len = len(rhats)
                if ep_len >= 3 and args.outcome_alpha and args.outcome_alpha > 0.0:
                    mean = sum(rhats) / ep_len
                    third = max(1, ep_len // 3)
                    early = sum(rhats[:third]) / third
                    mid = sum(rhats[third : 2 * third]) / third
                    late = sum(rhats[2 * third :]) / max(1, ep_len - 2 * third)
                    outcome_rhats[outcome].append((ep_len, mean, early, mid, late))
                per_lane_cur_rhats[i] = []
                obs, _info = envs[i].reset()
                raw = [float(x) for x in obs]
                raw_next[i] = raw
                next_lists[i] = normalize_obs(raw)
                agent.mark_boundary(i)

        obs_lists = next_lists

        if args.log_every and step > 0 and step % args.log_every == 0:
            diags = agent.diagnostics()
            # Aggregate recent returns across lanes.
            all_recent: list[float] = []
            for lane in ep_returns:
                all_recent.extend(lane[-5:])
            avg_return = sum(all_recent) / max(1, len(all_recent))

            def _safe(x, default=float("nan")):
                # serde_json serializes NaN/Inf as null — tolerate that in logs.
                return float(x) if x is not None else default

            def _mean(key):
                return sum(_safe(d[key]) for d in diags) / args.lanes

            wm = _safe(diags[0]["loss_world_model"])
            pi = _safe(diags[0]["loss_policy"])
            rew = _mean("reward_mean")
            sup = _mean("reward_surprise")
            hom = _mean("reward_homeo")
            ent = _mean("policy_entropy")
            elapsed = time.time() - t0
            sps = (step * args.lanes) / max(1e-3, elapsed)
            # L1 diagnostics: option distribution + goal distance.
            opt_counts: dict[int, int] = {}
            gdist = 0.0
            for d in diags:
                o = int(_safe(d.get("current_option", 0), 0))
                opt_counts[o] = opt_counts.get(o, 0) + 1
                gdist += _safe(d.get("goal_distance", 0.0), 0.0)
            gdist /= max(1, args.lanes)
            opt_str = "/".join(str(opt_counts.get(i, 0)) for i in range(max(opt_counts.keys()) + 1)) if opt_counts else "-"

            # Rolling soft-rate over the last N outcomes — this is the
            # "is the agent improving?" signal. Cumulative rate gets
            # pulled down by early poor episodes and hides a learning
            # curve.
            soft_in_window = sum(1 for o in recent_outcomes if o == "soft")
            window_size = len(recent_outcomes)
            soft_pct_window = 100.0 * soft_in_window / max(1, window_size)
            window_log.append((step, soft_in_window, window_size))

            r_hat_mean = sum(_safe(d.get("r_hat", 0.0), 0.0) for d in diags) / args.lanes
            outcome_base = _safe(diags[0].get("outcome_baseline", 0.0), 0.0)
            outcome_loss = _safe(diags[0].get("outcome_loss", 0.0), 0.0)
            print(
                f"step={step:>5} eps={total_episodes:>3} "
                f"avg_ret={avg_return:+7.1f} soft%(last{window_size})={soft_pct_window:4.1f} | "
                f"wm={wm:.3f} pi={pi:.3f} "
                f"r={rew:+6.3f} surp={sup:+5.2f} homeo={hom:+6.2f} "
                f"ent={ent:.2f} opts={opt_str} gdist={gdist:.2f} "
                f"m6:r_hat={r_hat_mean:+5.2f} base={outcome_base:+6.1f} loss={outcome_loss:.3f} "
                f"| {sps:5.1f} env-steps/s"
            )

    for env in envs:
        env.close()

    # Final per-lane summary. The metric that matters for "land safely,
    # anywhere" is `soft`: episodes that terminated with both legs touching
    # and low terminal speed. Crash = terminated fast / with no legs;
    # timeout = episode ran out of steps without resolving either way.
    print("\n--- per-lane episode returns ---")
    total_soft = 0
    total_crash = 0
    for i, lane in enumerate(ep_returns):
        if lane:
            outcomes = ep_outcomes[i]
            soft = outcomes.count("soft")
            crash = outcomes.count("crash")
            timeout = outcomes.count("timeout")
            total_soft += soft
            total_crash += crash
            print(
                f"  lane {i}: episodes={len(lane):>3} "
                f"soft={soft:>2} crash={crash:>2} timeout={timeout:>2} | "
                f"mean_ret={sum(lane) / len(lane):+7.1f} "
                f"best={max(lane):+7.1f}"
            )
        else:
            print(f"  lane {i}: no completed episodes")
    print(
        f"\nTotals: {total_soft} soft landings (anywhere), "
        f"{total_crash} crashes, {total_episodes} episodes"
    )

    # M6 mechanism check: did R̂ actually discriminate between
    # soft-landing and crash trajectories? This is what the principle
    # predicts — if it's silent here, the head isn't capturing what
    # we hoped it would, regardless of the cumulative-soft number.
    if args.outcome_alpha and args.outcome_alpha > 0.0:
        any_records = any(len(v) for v in outcome_rhats.values())
        if any_records:
            print("\n--- M6 R̂ discrimination (mean r_hat by outcome) ---")
            print(
                f"{'outcome':<8} {'episodes':>8} {'ep_len':>7} {'mean':>7} "
                f"{'early':>7} {'mid':>7} {'late':>7}"
            )
            summary = {}
            for outcome, records in outcome_rhats.items():
                if not records:
                    continue
                n = len(records)
                ep_len_m = sum(r[0] for r in records) / n
                m = sum(r[1] for r in records) / n
                e = sum(r[2] for r in records) / n
                mi = sum(r[3] for r in records) / n
                la = sum(r[4] for r in records) / n
                summary[outcome] = (n, ep_len_m, m, e, mi, la)
                print(
                    f"{outcome:<8} {n:>8d} {ep_len_m:>7.1f} {m:>+7.2f} "
                    f"{e:>+7.2f} {mi:>+7.2f} {la:>+7.2f}"
                )
            if "soft" in summary and "crash" in summary:
                s_mean = summary["soft"][2]
                c_mean = summary["crash"][2]
                print(
                    f"\n  soft − crash gap: mean={s_mean - c_mean:+.3f}  "
                    f"early={summary['soft'][3] - summary['crash'][3]:+.3f}  "
                    f"late={summary['soft'][5] - summary['crash'][5]:+.3f}"
                )
                verdict = (
                    "discriminating — M6 head encodes trajectory quality"
                    if s_mean > c_mean + 0.2
                    else "silent or inverted — head not capturing quality"
                )
                print(f"  verdict: {verdict}")

    diags = agent.diagnostics()
    print("\n--- final diagnostics ---")
    for i, d in enumerate(diags):
        print(
            f"  lane {i}: wm={d['loss_world_model']:.3f} "
            f"pi={d['loss_policy']:.3f} "
            f"r={d['reward_mean']:+6.3f} "
            f"surp={d['reward_surprise']:+5.2f} "
            f"novelty={d['reward_novelty']:+5.2f} "
            f"homeo={d['reward_homeo']:+6.2f} "
            f"order={d['reward_order']:+5.2f} "
            f"ent={d['policy_entropy']:.2f} "
            f"buf={d['buffer_len']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

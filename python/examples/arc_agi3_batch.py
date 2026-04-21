"""First test of kindle on ARC-AGI-3 (2026-04-20).

ARC-AGI-3 is an interactive-reasoning benchmark: each env is a
turn-based game with no instructions, no descriptions, no stated
win condition. Agent sees a 64×64 grid of int8 values (0–15
colour channels), picks one of up to 7 discrete actions plus
optional coordinate-parameterized actions, and has to figure
out the game from observation.

From the M7 closeout: kindle's reward class cannot express
event-ordered terminal success (the LunarLander finding). ARC-
AGI-3 games are *structurally* the same shape of task —
level-completion is an event, not a steady-state homeo condition
— so we predict kindle won't solve them without external
supervision. But we don't know in advance what signal from the
game IS self-observable and usable. This script is a first
exploration.

Setup:
  - One local env (no API key needed; arc_agi auto-gets an
    anonymous key via Arcade).
  - 64×64 int8 frame → downsample to 8×8 via average pool →
    flatten to 64-dim obs (matches kindle's OBS_TOKEN_DIM=64).
  - Simple discrete actions only (ignore complex / coord
    actions for v1). Kindle emits MAX_ACTION_DIM=6 action dims;
    we map to the game's `available_actions` list.
  - Homeo signal: `levels_completed` increment (rare event)
    plus a frame-entropy term (count of unique cells, which on
    ARC-AGI-3 correlates with "discovering states"). No
    landing-specific hand-coding.
"""

from __future__ import annotations

import argparse
import math
import sys
import time


def main() -> int:
    try:
        from arc_agi import Arcade
        from arcengine import GameAction, GameState
    except ImportError as exc:
        print(f"missing arc_agi toolkit: {exc}", file=sys.stderr)
        return 1
    import numpy as np
    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="ls20",
                        help="ARC-AGI-3 game id prefix (e.g. 'ls20', 'ar25').")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=400,
                        help="Cap on actions per ARC episode before forced reset.")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--agent", choices=["kindle", "random"], default="kindle")
    parser.add_argument("--entropy-floor", type=float, default=0.0)
    parser.add_argument("--advantage-clamp", type=float, default=20.0)
    parser.add_argument("--watchdog-threshold", type=float, default=1e6)
    parser.add_argument("--levels-reward-scale", type=float, default=10.0,
                        help="Homeo spike magnitude per level completed.")
    parser.add_argument("--encoder", choices=["mlp", "cnn"], default="mlp",
                        help="'mlp' (default) = pre-pooled 64-dim token → MLP encoder. "
                        "'cnn' = raw 64×64 grid → conv encoder (spatial info preserved).")
    parser.add_argument("--reward-surprise", type=float, default=None,
                        help="Weight on the world-model surprise primitive. Default 1.0 "
                        "from kindle. For ARC, try 5-10 (homeo is meaningless here, so "
                        "surprise becomes the primary exploration signal).")
    parser.add_argument("--reward-novelty", type=float, default=None,
                        help="Weight on the latent-visit-count novelty primitive. "
                        "Default 0.5.")
    parser.add_argument("--reward-homeostatic", type=float, default=None,
                        help="Weight on the homeostatic primitive. Default 2.0 from "
                        "kindle. For ARC, try 0.1 — the only 'homeo' signal we have "
                        "here is the levels-completed delta, which is sparse.")
    parser.add_argument("--reward-order", type=float, default=None,
                        help="Weight on the order primitive. Default 0.5. Reduced for "
                        "CNN mode where the order-digest may be less meaningful.")
    parser.add_argument("--grid-resolution", type=float, default=None,
                        help="Latent-grid bucket size for the novelty visit counter. "
                        "Default 0.5. Lower = finer buckets = novelty stays high longer "
                        "but fewer rare-state matches.")
    parser.add_argument("--rnd-alpha", type=float, default=None,
                        help="RND curiosity weight. 0 (default) disables. Try 0.5-5.0 on "
                        "visual envs where kindle's surprise/novelty primitives saturate.")
    parser.add_argument("--rnd-feature-dim", type=int, default=None,
                        help="RND target/predictor output size. Default 16.")
    parser.add_argument("--rnd-hidden-dim", type=int, default=None,
                        help="RND MLP hidden-layer width. Default 64.")
    parser.add_argument("--rnd-lr", type=float, default=None,
                        help="RND predictor LR. Default = learning_rate × 0.3.")
    parser.add_argument("--rnd-reset-on-level", action="store_true",
                        help="Reset the RND predictor each time levels_completed "
                        "increases. Re-activates curiosity when the agent reaches a new "
                        "level whose state distribution the old predictor has already fit.")
    parser.add_argument("--coord-alpha", type=float, default=None,
                        help="Coord action head weight (REINFORCE scale). 0 (default) "
                        "disables and the harness falls back to random coords for "
                        "complex actions. >0 lets kindle learn the spatial policy.")
    parser.add_argument("--coord-sigma", type=float, default=None,
                        help="Gaussian exploration noise for coord head. Default 0.3.")
    parser.add_argument("--coord-lr", type=float, default=None,
                        help="Coord head LR. Default = learning_rate × 0.3.")
    parser.add_argument("--delta-goal-alpha", type=float, default=None,
                        help="M8 delta-goal reward weight. 0 (default) disables. "
                        "Try 0.1-1.0 on envs where state-change events are rare "
                        "but reaching them is the goal. Self-supervised — no "
                        "task-specific signal.")
    parser.add_argument("--delta-goal-threshold", type=float, default=None,
                        help="Min per-step latent-delta to register a new M8 goal. "
                        "Default 0.5.")
    parser.add_argument("--delta-goal-merge-radius", type=float, default=None,
                        help="M8 goals within this L2 distance of an existing goal "
                        "are dropped as duplicates. Default 0.1.")
    parser.add_argument("--delta-goal-bank-size", type=int, default=None,
                        help="M8 max goal-bank size (oldest evicted). Default 64.")
    parser.add_argument("--delta-goal-distance-clamp", type=float, default=None,
                        help="M8 symmetric distance clamp to bound per-step bonus "
                        "magnitude (pre-alpha). Default 5.0.")
    parser.add_argument("--delta-goal-surprise-threshold", type=float, default=None,
                        help="M8 v2: min WM pred_error to enable goal recording "
                        "this step. Default 0.5. Banks surprising transitions "
                        "(WM didn't predict them) rather than routine ones.")
    parser.add_argument("--xeps-alpha", type=float, default=None,
                        help="Cross-episode state-action novelty weight. 0 "
                        "(default) disables. Try 0.5-2.0 on games where the "
                        "agent reaches a local optimum every episode and "
                        "retries the same actions there. Self-supervised.")
    parser.add_argument("--xeps-grid-resolution", type=float, default=None,
                        help="Grid bucket size for xeps state key. "
                        "Default = main grid_resolution.")
    args = parser.parse_args()

    # --- Set up the local ARC-AGI-3 env ---
    arcade = Arcade()
    envs_info = arcade.get_environments()
    env_info = next(
        (e for e in envs_info if e.game_id.startswith(args.game)), None
    )
    if env_info is None:
        print(f"no env matching prefix {args.game!r}", file=sys.stderr)
        print(
            "available:", sorted(e.game_id[:4] for e in envs_info), file=sys.stderr
        )
        return 1
    env = arcade.make(env_info.game_id)
    obs_raw = env.reset()
    frame = np.asarray(obs_raw.frame[0], dtype=np.float32)  # (64, 64)
    print(
        f"game={env_info.title} id={env_info.game_id} "
        f"available_actions={obs_raw.available_actions} "
        f"win_levels={obs_raw.win_levels} "
        f"frame_shape={frame.shape}"
    )

    available_actions = list(obs_raw.available_actions)
    if not available_actions:
        print("no available actions on initial frame", file=sys.stderr)
        return 1
    # kindle's discrete adapter emits actions 0..num_actions-1.
    # We use num_actions = min(6, len(available_actions)) so
    # kindle's MAX_ACTION_DIM=6 covers the space. (Complex actions
    # — ACTION6 — are allowed here; their (x, y) payload is filled
    # with random coords per step by `action_to_game` below.)
    num_actions = min(6, len(available_actions))

    def preprocess(frame_ndarray: np.ndarray) -> list[float]:
        """64×64 int → 8×8 mean-pooled → flat 64-dim float in [0, 1].
        The pooled token still feeds kindle's reward circuit (order
        digest) and is stored on Transition for replay."""
        arr = frame_ndarray.astype(np.float32) / 15.0  # ARC colours 0..15
        pooled = arr.reshape(8, 8, 8, 8).mean(axis=(1, 3))
        return pooled.flatten().tolist()

    def preprocess_visual(frame_ndarray: np.ndarray) -> list[float]:
        """64×64 int → flat 4096-dim float in [0, 1]. Fed to the CNN
        encoder directly when `--encoder cnn`."""
        return (frame_ndarray.astype(np.float32) / 15.0).flatten().tolist()

    # Pre-build a value→member map; `GameAction(value)` is broken
    # on this enum type in the installed arcengine build.
    action_by_value = {int(a.value): a for a in GameAction}

    def action_to_game(
        kindle_action_idx: int,
        kindle_xy: tuple[float, float] | None = None,
    ) -> GameAction:
        """Map kindle's 0-indexed discrete output to the current
        frame's `available_actions`. For complex actions
        (currently ACTION6), attach `(x, y)` coordinates.

        If `kindle_xy` is supplied (values in `[-1, 1]` from the
        coord head), rescale to `[0, 63]`; otherwise fall back to
        uniform random coords in `[0, 63]`. Simple actions
        ignore the coordinate payload.
        """
        aa = available_actions
        if not aa:
            return action_by_value[1]
        idx = max(0, min(kindle_action_idx, len(aa) - 1))
        action_num = int(aa[idx])
        a = action_by_value[action_num]
        if a.is_complex():
            if kindle_xy is not None:
                # [-1, 1] → [0, 63].
                sx, sy = kindle_xy
                x = int(round((sx + 1.0) * 0.5 * 63.0))
                y = int(round((sy + 1.0) * 0.5 * 63.0))
                x = max(0, min(63, x))
                y = max(0, min(63, y))
            else:
                x = rng.randrange(64)
                y = rng.randrange(64)
            a.set_data({"x": x, "y": y})
        return a

    # --- Kindle agent (one lane) ---
    if args.agent == "kindle":
        obs_dim = 64  # token dim for reward circuit (always 64)
        agent_kwargs = dict(
            obs_dim=obs_dim,
            num_actions=num_actions,
            batch_size=1,
            env_ids=[0],
            seed=args.seed,
            learning_rate=args.lr,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            entropy_floor=args.entropy_floor,
            advantage_clamp=args.advantage_clamp,
            policy_loss_watchdog_threshold=args.watchdog_threshold,
        )
        if args.encoder == "cnn":
            agent_kwargs.update(
                encoder_kind="cnn",
                encoder_channels=1,
                encoder_height=64,
                encoder_width=64,
            )
        if args.reward_surprise is not None:
            agent_kwargs["reward_surprise"] = args.reward_surprise
        if args.reward_novelty is not None:
            agent_kwargs["reward_novelty"] = args.reward_novelty
        if args.reward_homeostatic is not None:
            agent_kwargs["reward_homeostatic"] = args.reward_homeostatic
        if args.reward_order is not None:
            agent_kwargs["reward_order"] = args.reward_order
        if args.grid_resolution is not None:
            agent_kwargs["grid_resolution"] = args.grid_resolution
        if args.rnd_alpha is not None:
            agent_kwargs["rnd_reward_alpha"] = args.rnd_alpha
        if args.rnd_feature_dim is not None:
            agent_kwargs["rnd_feature_dim"] = args.rnd_feature_dim
        if args.rnd_hidden_dim is not None:
            agent_kwargs["rnd_hidden_dim"] = args.rnd_hidden_dim
        if args.rnd_lr is not None:
            agent_kwargs["rnd_lr"] = args.rnd_lr
        if args.coord_alpha is not None:
            agent_kwargs["coord_action_alpha"] = args.coord_alpha
        if args.coord_sigma is not None:
            agent_kwargs["coord_sigma"] = args.coord_sigma
        if args.coord_lr is not None:
            agent_kwargs["coord_lr"] = args.coord_lr
        if args.delta_goal_alpha is not None:
            agent_kwargs["delta_goal_alpha"] = args.delta_goal_alpha
        if args.delta_goal_threshold is not None:
            agent_kwargs["delta_goal_threshold"] = args.delta_goal_threshold
        if args.delta_goal_merge_radius is not None:
            agent_kwargs["delta_goal_merge_radius"] = args.delta_goal_merge_radius
        if args.delta_goal_bank_size is not None:
            agent_kwargs["delta_goal_bank_size"] = args.delta_goal_bank_size
        if args.delta_goal_distance_clamp is not None:
            agent_kwargs["delta_goal_distance_clamp"] = args.delta_goal_distance_clamp
        if args.delta_goal_surprise_threshold is not None:
            agent_kwargs["delta_goal_surprise_threshold"] = args.delta_goal_surprise_threshold
        if args.xeps_alpha is not None:
            agent_kwargs["xeps_reward_alpha"] = args.xeps_alpha
        if args.xeps_grid_resolution is not None:
            agent_kwargs["xeps_grid_resolution"] = args.xeps_grid_resolution
        agent = kindle.BatchAgent(**agent_kwargs)
    else:
        agent = None  # random baseline

    import random
    rng = random.Random(args.seed)

    current_obs = preprocess(frame)
    last_levels = int(obs_raw.levels_completed)
    ep_step = 0
    ep_count = 0
    ep_levels_at_end: list[int] = []
    ep_lens: list[int] = []
    levels_events = 0  # total level completions across all episodes
    t0 = time.time()

    def homeo_for(
        frame_arr: np.ndarray,
        new_levels: int,
        win_levels: int,
    ) -> list[dict]:
        """Kindle homeo-variable list. Two terms:

        - **distance-to-win**: `value = (win_levels − new_levels) ·
          scale`, target 0, tol 0. Kindle's homeo function returns
          `−|value − target|` as the contribution, so the reward
          is `−(win_levels − levels_completed) · scale`. This is
          a persistent negative signal that decreases in magnitude
          as levels are completed. At level-up, per-step reward
          jumps by `+scale` (the homeo weight multiplies it in
          `reward_circuit.compute`). Unlike the original
          `-delta_levels` formulation, this is a *sustained*
          incentive to progress, not a one-step spike at the
          exact moment of level completion.

        - **frame entropy**: `1 − unique_cells/16`, target 0, tol
          0.1. Small exploration nudge — higher when the frame is
          monotonous, zero when diverse.
        """
        remaining = max(0, win_levels - new_levels)
        levels_term = {
            "value": float(remaining) * args.levels_reward_scale,
            "target": 0.0,
            "tolerance": 0.0,
        }
        uniq = float(np.unique(frame_arr).size) / 16.0
        entropy_term = {
            "value": 1.0 - uniq,
            "target": 0.0,
            "tolerance": 0.1,
        }
        return [levels_term, entropy_term]

    for step in range(args.steps):
        state = obs_raw.state
        need_reset = state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or (
            state is GameState.WIN
        )
        if need_reset or ep_step >= args.max_episode_steps:
            if ep_step > 0:
                ep_count += 1
                ep_levels_at_end.append(int(obs_raw.levels_completed))
                ep_lens.append(ep_step)
            obs_raw = env.reset()
            frame = np.asarray(obs_raw.frame[0], dtype=np.float32)
            current_obs = preprocess(frame)
            last_levels = int(obs_raw.levels_completed)
            available_actions[:] = list(obs_raw.available_actions) or available_actions
            ep_step = 0
            if agent is not None:
                agent.mark_boundary(0)

        # Choose action
        if agent is not None:
            if args.encoder == "cnn":
                agent.set_visual_obs(preprocess_visual(frame))
            # Sample coords BEFORE act so kindle can cache the
            # sample state; train happens post-observe.
            kindle_xy = None
            if args.coord_alpha is not None and args.coord_alpha > 0:
                kindle_xy = tuple(agent.sample_coords()[0])
            actions = agent.act([current_obs])
            kindle_action = int(actions[0])
        else:
            kindle_action = rng.randrange(num_actions)
            kindle_xy = None
        game_action = action_to_game(kindle_action, kindle_xy)

        # Step env
        obs_raw = env.step(game_action)
        frame = np.asarray(obs_raw.frame[0], dtype=np.float32)
        if list(obs_raw.available_actions):
            available_actions = list(obs_raw.available_actions)
        new_obs = preprocess(frame)

        new_levels = int(obs_raw.levels_completed)
        delta_levels = new_levels - last_levels
        if delta_levels > 0:
            levels_events += delta_levels
        last_levels = new_levels

        # Feed observation back to kindle
        if agent is not None:
            if args.encoder == "cnn":
                agent.set_visual_obs(preprocess_visual(frame))
            homeos = [homeo_for(frame, new_levels, int(obs_raw.win_levels))]
            agent.observe([new_obs], [kindle_action], homeostatic=homeos)
            # Coord-head REINFORCE step using this step's reward.
            if args.coord_alpha is not None and args.coord_alpha > 0:
                agent.train_coord_head()
            # Reset RND predictor on level transitions to
            # re-activate curiosity on the new state distribution.
            if args.rnd_reset_on_level and delta_levels > 0:
                agent.reset_rnd_predictor()

        current_obs = new_obs
        ep_step += 1

        if args.log_every and step > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step / max(1e-3, elapsed)
            last_ent = ""
            if agent is not None:
                diags = agent.diagnostics()
                d = diags[0]
                dg_bank = agent.delta_goal_bank_size()
                xeps_pairs = agent.xeps_distinct_pairs()
                last_ent = (
                    f"| wm={float(d['loss_world_model']):.3f} "
                    f"pi={float(d['loss_policy']):.3f} "
                    f"ent={float(d['policy_entropy']):.2f} "
                    f"surp={float(d['reward_surprise']):+5.2f} "
                    f"nov={float(d['reward_novelty']):+4.2f} "
                    f"hom={float(d['reward_homeo']):+5.2f} "
                    f"ord={float(d['reward_order']):+4.2f} "
                    f"rnd={float(d.get('rnd_mse', 0.0)):+5.2f} "
                    f"dg={dg_bank:>3} xeps={xeps_pairs:>4}"
                )
            print(
                f"step={step:>5} eps={ep_count:>3} lvl_events={levels_events:>3} "
                f"cur_lvl={new_levels} avail={available_actions} "
                f"{last_ent} | {sps:5.1f} steps/s"
            )

    # Final summary
    elapsed = time.time() - t0
    sps = args.steps / max(1e-3, elapsed)
    print()
    print(f"--- {env_info.title} summary ({args.agent}) ---")
    print(f"total steps: {args.steps}")
    print(f"episodes completed: {ep_count}")
    print(f"total level events: {levels_events}")
    if ep_levels_at_end:
        mean_final = sum(ep_levels_at_end) / len(ep_levels_at_end)
        print(
            f"mean levels at episode end: {mean_final:.2f} "
            f"(max {max(ep_levels_at_end)} / win {obs_raw.win_levels})"
        )
        mean_len = sum(ep_lens) / len(ep_lens)
        print(f"mean episode length: {mean_len:.1f}")
    print(f"throughput: {sps:.1f} steps/s ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

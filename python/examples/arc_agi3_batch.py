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
    # kindle's MAX_ACTION_DIM=6 covers the space.
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
    # Filter to simple actions only (ACTION1..ACTION7). Complex
    # (coordinate-parameterized) actions need {x, y} data we
    # don't synthesize in this adapter.
    SIMPLE_ACTIONS = set(range(1, 8))

    def action_to_game(kindle_action_idx: int) -> GameAction:
        """Map kindle's 0-indexed discrete output to the current
        frame's simple available_actions. If kindle picks an
        index outside that list, clip to the last available."""
        aa = [v for v in available_actions if int(v) in SIMPLE_ACTIONS]
        if not aa:
            # Fall back to ACTION1; games with only complex
            # actions are out of scope for this adapter.
            return action_by_value[1]
        idx = max(0, min(kindle_action_idx, len(aa) - 1))
        action_num = int(aa[idx])
        return action_by_value[action_num]

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

    def homeo_for(frame_arr: np.ndarray, delta_levels: int) -> list[dict]:
        """Kindle homeo-variable list. Two terms:
          - levels progress spike: value = -delta_levels·scale
            (negative of a negative deviation, i.e. progress is
            reward). Target 0, tol 0.
          - frame entropy: unique-values proxy — encourages
            encountering new frame patterns."""
        levels_term = {
            "value": -delta_levels * args.levels_reward_scale,
            "target": 0.0,
            "tolerance": 0.0,
        }
        # Small frame-entropy homeo (number of unique cell colours)
        # to give kindle a scalar that meaningfully varies with
        # exploration. Normalize by 16 colours.
        uniq = float(np.unique(frame_arr).size) / 16.0
        entropy_term = {
            "value": 1.0 - uniq,  # high when obs is monotonous
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
            actions = agent.act([current_obs])
            kindle_action = int(actions[0])
        else:
            kindle_action = rng.randrange(num_actions)
        game_action = action_to_game(kindle_action)

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
            homeos = [homeo_for(frame, delta_levels)]
            agent.observe([new_obs], [kindle_action], homeostatic=homeos)

        current_obs = new_obs
        ep_step += 1

        if args.log_every and step > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step / max(1e-3, elapsed)
            last_ent = ""
            if agent is not None:
                diags = agent.diagnostics()
                d = diags[0]
                last_ent = (
                    f"| wm={float(d['loss_world_model']):.3f} "
                    f"pi={float(d['loss_policy']):.3f} "
                    f"ent={float(d['policy_entropy']):.2f}"
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

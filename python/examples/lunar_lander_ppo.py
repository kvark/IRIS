"""PPO ceiling test on kindle's intrinsic reward for LunarLander.

Wraps `gymnasium`'s LunarLander so the reward fed to the policy is
kindle's homeostatic reward (the sum over the v3 shaping homeo
variables with kindle's default weight `homeostatic = 2.0`), not
the env's native reward. Lets us ask: **can a well-tuned standard
policy-gradient optimizer (PPO) converge on a landing policy when
the reward signal is the same one kindle has been training on?**

If PPO lands → kindle's reward shape is fine and the plateau we
observed is kindle's optimizer stack (supports the "advantage clamp
is the commitment suppressor" ablation finding).

If PPO fails to land → the reward signal itself doesn't contain a
gradient that leads to landing under local search. Then even a
perfect optimizer couldn't solve it from this reward.

Usage:
    python lunar_lander_ppo.py --steps 100000 --seed 42
    python lunar_lander_ppo.py --shaping v5 --reward-mix homeo_only
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# --- Kindle homeo shapings (copied from lunar_lander_batch.py to
#     keep this script standalone) ---

_FUEL_COST = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}


def lunar_lander_homeo_v3(obs, action):
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


def lunar_lander_homeo_v5(obs, action):
    entries = lunar_lander_homeo_v3(obs, action)
    # v5: amplified not_safely_landed (last entry); replace its value.
    entries[-1] = {
        "value": entries[-1]["value"] * 50.0,
        "target": 0.0,
        "tolerance": 0.0,
    }
    return entries


def lunar_lander_homeo_v6(obs, action):
    """v6: reward weights tuned for landing under local policy search.

    v3 and v5 both fail PPO's ceiling test because their reward
    landscape has 'terminate fast' as the global/local optimum —
    every per-step homeo term is ≥ 0 deviation, so every moment
    aloft accumulates negative reward. Fast crash < hover < land
    in cumulative cost, so the policy converges on the cheapest
    one.

    v6 restructures to make controlled-flight states *less bad*
    and dangerous states *much worse*:

      - altitude weight = 0: no penalty for being aloft. Long
        controlled flights aren't uniformly worse than short
        crashes just because they're longer.
      - crash_risk × 50 (was 10): fast-descent-near-ground is
        heavily punished. Landing requires slowing before ground
        contact.
      - speed × proximity × 20 (was 1): same — high speed near
        ground is landing's specific failure mode.
      - fuel × 0.1 (was 1): thrust is near-free. Don't punish the
        deceleration actions needed to land.
      - angle × proximity × 5 (was 1): upright terminal pose
        matters.
      - not_safely_landed × 0.5 (was 1): reduce the
        "every-step-not-landed" penalty — this is what made
        longer-controlled-descent episodes look worse than
        crashes under v5.
      - vy keeps target -0.25 (controlled descent rate).
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
        {"value": crash_risk * 50.0, "target": 0.0, "tolerance": 0.0},
        {"value": abs(angle) * proximity * 5.0, "target": 0.0, "tolerance": 0.05},
        {"value": speed * proximity * 20.0, "target": 0.0, "tolerance": 0.1},
        {"value": fuel * 0.1, "target": 0.0, "tolerance": 0.0},
        # altitude weight = 0: term is gone entirely.
        {"value": vy, "target": -0.25, "tolerance": 0.25},
        {"value": not_safely_landed * 0.5, "target": 0.0, "tolerance": 0.0},
    ]


_SHAPINGS = {
    "v3": lunar_lander_homeo_v3,
    "v5": lunar_lander_homeo_v5,
    "v6": lunar_lander_homeo_v6,
}


# --- Kindle-equivalent homeo reward reduction ---

def kindle_homeo_penalty(entries, weight: float = 2.0) -> float:
    """Replicate `reward::RewardCircuit::homeostatic` — negative of
    summed out-of-tolerance deviation, scaled by the homeostatic
    weight. This is the `homeo` component of kindle's reward; we
    multiply by the same `2.0` kindle uses so magnitudes match.
    """
    penalty = 0.0
    for v in entries:
        dev = abs(v["value"] - v["target"]) - v["tolerance"]
        if dev > 0:
            penalty += dev
    return -weight * penalty


# --- Obs normalization (same scales as lunar_lander_batch.py) ---

_LL_SCALE = [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0]


def normalize_obs(obs):
    return np.array([float(x) / s for x, s in zip(obs, _LL_SCALE)], dtype=np.float32)


class KindleRewardWrapper(gym.Wrapper):
    """Substitutes the env's reward with kindle's homeo penalty.

    Also classifies each termination as soft/crash/timeout using the
    same logic as `lunar_lander_batch.py` so we can report the
    primary metric (cumulative soft-rate) consistently.

    `shaping == 'native'` disables substitution entirely and passes
    the env's native reward through — a control to verify that this
    harness + sb3 PPO does land LunarLander when given the canonical
    reward (since kindle-reward shapes seem to universally fail,
    this is our ground-truth that PPO works).
    """

    def __init__(
        self,
        env,
        shaping: str = "v3",
        homeo_weight: float = 2.0,
        terminal_soft_bonus: float = 0.0,
        terminal_crash_penalty: float = 0.0,
    ):
        super().__init__(env)
        self.shaping = shaping
        self.shaping_fn = _SHAPINGS[shaping] if shaping != "native" else None
        self.homeo_weight = homeo_weight
        self.terminal_soft_bonus = terminal_soft_bonus
        self.terminal_crash_penalty = terminal_crash_penalty
        # Override observation space to the normalized form.
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(8,), dtype=np.float32
        )
        self._last_raw = None
        self._ep_outcomes: list[str] = []
        self._ep_len = 0
        self._ep_return = 0.0  # kindle-reward accumulated

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self._last_raw = raw_obs
        self._ep_len = 0
        self._ep_return = 0.0
        return normalize_obs(raw_obs), info

    def step(self, action):
        raw_next, env_reward, terminated, truncated, info = self.env.step(action)
        if self.shaping == "native":
            r = float(env_reward)
        else:
            entries = self.shaping_fn(raw_next, action)
            r = kindle_homeo_penalty(entries, self.homeo_weight)

        done = terminated or truncated
        if done:
            leg1, leg2 = float(raw_next[6]), float(raw_next[7])
            vx, vy = float(raw_next[2]), float(raw_next[3])
            term_speed = math.sqrt(vx * vx + vy * vy)
            if terminated and leg1 > 0 and leg2 > 0 and term_speed < 0.25:
                outcome = "soft"
                r += self.terminal_soft_bonus
            elif terminated:
                outcome = "crash"
                r -= self.terminal_crash_penalty
            else:
                outcome = "timeout"

        self._ep_len += 1
        self._ep_return += r

        if done:
            self._ep_outcomes.append(outcome)
            info["ll_outcome"] = outcome
            info["ll_ep_len"] = self._ep_len
            info["ll_ep_kindle_return"] = self._ep_return

        self._last_raw = raw_next
        return normalize_obs(raw_next), float(r), bool(terminated), bool(truncated), info


class OutcomeTrackingCallback(BaseCallback):
    """Aggregates outcome counts from the wrapped env and prints a
    rolling soft-rate every `log_every` steps."""

    def __init__(self, log_every: int):
        super().__init__()
        self.log_every = log_every
        self.outcomes: list[str] = []
        self.ep_lens: list[int] = []
        self.ep_kindle_returns: list[float] = []
        self._last_log = 0
        self._t0 = None

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        # Harvest terminal infos from wrapped env(s)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "ll_outcome" in info:
                self.outcomes.append(info["ll_outcome"])
                self.ep_lens.append(info["ll_ep_len"])
                self.ep_kindle_returns.append(info["ll_ep_kindle_return"])
        # Log rolling soft-rate
        if self.num_timesteps - self._last_log >= self.log_every:
            self._last_log = self.num_timesteps
            window = 200
            recent = self.outcomes[-window:]
            soft = sum(1 for o in recent if o == "soft")
            total = len(self.outcomes)
            cum_soft = sum(1 for o in self.outcomes if o == "soft")
            cum_pct = 100.0 * cum_soft / max(1, total)
            window_pct = 100.0 * soft / max(1, len(recent))
            # Recent kindle-reward mean
            recent_ret = self.ep_kindle_returns[-window:]
            mean_ret = (
                sum(recent_ret) / max(1, len(recent_ret))
                if recent_ret
                else 0.0
            )
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / max(1e-3, elapsed)
            print(
                f"step={self.num_timesteps:>6} eps={total:>4} "
                f"cum_soft%={cum_pct:5.2f} "
                f"window_soft%(last{len(recent)})={window_pct:5.1f} "
                f"mean_kindle_ret(last200)={mean_ret:+7.1f} "
                f"| {sps:5.0f} steps/s"
            )
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--shaping", choices=list(_SHAPINGS.keys()) + ["native"], default="v3")
    parser.add_argument("--homeo-weight", type=float, default=2.0)
    parser.add_argument("--terminal-soft-bonus", type=float, default=0.0,
                        help="Once-only reward added at a soft-landing terminal.")
    parser.add_argument("--terminal-crash-penalty", type=float, default=0.0,
                        help="Once-only penalty subtracted at a crash terminal.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length per env per update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="PPO entropy regularization coefficient")
    parser.add_argument("--policy-arch", default="64,64",
                        help="comma-separated hidden layer sizes")
    args = parser.parse_args()

    print(
        f"env={args.env} shaping={args.shaping} homeo_weight={args.homeo_weight} "
        f"steps={args.steps} seed={args.seed}"
    )

    env = gym.make(args.env)
    env = KindleRewardWrapper(
        env,
        shaping=args.shaping,
        homeo_weight=args.homeo_weight,
        terminal_soft_bonus=args.terminal_soft_bonus,
        terminal_crash_penalty=args.terminal_crash_penalty,
    )

    hidden = [int(x) for x in args.policy_arch.split(",")]
    policy_kwargs = dict(net_arch=hidden)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        verbose=0,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
    )

    cb = OutcomeTrackingCallback(log_every=args.log_every)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)

    # Final summary
    total = len(cb.outcomes)
    cum_soft = sum(1 for o in cb.outcomes if o == "soft")
    cum_crash = sum(1 for o in cb.outcomes if o == "crash")
    cum_timeout = sum(1 for o in cb.outcomes if o == "timeout")
    print(
        f"\nTotals: {cum_soft} soft / {cum_crash} crash / {cum_timeout} timeout "
        f"over {total} episodes"
    )
    if total:
        print(
            f"  cumulative soft-rate: {100.0 * cum_soft / total:.2f}%"
        )
        # Last-third soft-rate shows trained-policy behavior
        tail = cb.outcomes[2 * total // 3 :]
        tail_soft = sum(1 for o in tail if o == "soft")
        print(
            f"  last-third soft-rate: {100.0 * tail_soft / max(1, len(tail)):.2f}% "
            f"({tail_soft} / {len(tail)})"
        )
        mean_len = sum(cb.ep_lens) / max(1, len(cb.ep_lens))
        tail_mean_len = sum(cb.ep_lens[2 * total // 3 :]) / max(
            1, len(cb.ep_lens[2 * total // 3 :])
        )
        print(
            f"  episode length: mean={mean_len:.1f}, last-third mean={tail_mean_len:.1f}"
        )
        mean_ret = sum(cb.ep_kindle_returns) / max(1, len(cb.ep_kindle_returns))
        tail_ret = cb.ep_kindle_returns[2 * total // 3 :]
        tail_mean_ret = sum(tail_ret) / max(1, len(tail_ret))
        print(
            f"  kindle episode return: mean={mean_ret:+.1f}, "
            f"last-third mean={tail_mean_ret:+.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal A2C reference implementation in pure numpy + gymnasium.

This is the "control experiment" for the kindle CartPole investigation:
if standard A2C on this same machine + same env-step budget actually
reaches the +195 solve threshold, the bottleneck is definitively in
how kindle composes its training loop, not in meganeura primitives or
in CartPole being inherently hard at small budgets.

Architecture: 4 → 64 → 64 (tanh) → {logits[2], value}
Loss: -log π(a|s) · Â + 0.5 · MSE(V, R) - 0.01 · H(π)
Update: Adam, n_steps=5 rollout × 8 envs, GAE(0.95), γ=0.99.
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import gymnasium as gym


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class A2C:
    def __init__(self, obs_dim, n_actions, hidden=64, lr=7e-4, seed=0):
        rng = np.random.default_rng(seed)
        # Xavier-init weights, zero biases.
        def w(fan_in, fan_out):
            s = np.sqrt(2.0 / fan_in)
            return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * s
        self.W1 = w(obs_dim, hidden)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = w(hidden, hidden)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.Wp = w(hidden, n_actions)
        self.bp = np.zeros(n_actions, dtype=np.float32)
        self.Wv = w(hidden, 1)
        self.bv = np.zeros(1, dtype=np.float32)
        self.lr = lr
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "Wp": self.Wp, "bp": self.bp,
            "Wv": self.Wv, "bv": self.bv,
        }

    def forward(self, obs):
        # obs: [B, obs_dim]
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.Wp + self.bp
        value = (h2 @ self.Wv + self.bv).squeeze(-1)
        return logits, value, (obs, h1, h2)

    def backward(self, cache, dlogits, dvalue):
        # dlogits: [B, K], dvalue: [B]
        obs, h1, h2 = cache
        B = obs.shape[0]
        # value head
        dWv = h2.T @ dvalue.reshape(-1, 1) / B
        dbv = dvalue.mean(axis=0, keepdims=True).flatten()
        dh2 = dvalue.reshape(-1, 1) @ self.Wv.T
        # policy head
        dWp = h2.T @ dlogits / B
        dbp = dlogits.mean(axis=0)
        dh2 = dh2 + dlogits @ self.Wp.T
        # h2 = tanh(h1 W2 + b2)
        dpre2 = dh2 * (1.0 - h2 ** 2)
        dW2 = h1.T @ dpre2 / B
        db2 = dpre2.mean(axis=0)
        dh1 = dpre2 @ self.W2.T
        # h1 = tanh(obs W1 + b1)
        dpre1 = dh1 * (1.0 - h1 ** 2)
        dW1 = obs.T @ dpre1 / B
        db1 = dpre1.mean(axis=0)
        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "Wp": dWp, "bp": dbp,
            "Wv": dWv, "bv": dbv,
        }

    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        bc1 = 1 - beta1 ** self.t
        bc2 = 1 - beta2 ** self.t
        for k, p in self.params().items():
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * g * g
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=2000)
    args = p.parse_args()

    envs = gym.vector.SyncVectorEnv([lambda i=i: gym.make(args.env) for i in range(args.n_envs)])
    obs, _ = envs.reset(seed=args.seed)
    obs_dim = obs.shape[-1]
    n_actions = int(envs.single_action_space.n)

    agent = A2C(obs_dim, n_actions, lr=args.lr, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    ep_returns = [[] for _ in range(args.n_envs)]
    cur_ret = np.zeros(args.n_envs, dtype=np.float64)
    n_updates = args.total_steps // (args.n_envs * args.n_steps)
    env_steps = 0
    t0 = time.time()

    for upd in range(n_updates):
        # Rollout buffers
        obs_buf = np.zeros((args.n_steps, args.n_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.int64)
        rew_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
        val_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
        done_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)
        logp_buf = np.zeros((args.n_steps, args.n_envs), dtype=np.float32)

        # Collect rollout under FIXED policy
        for t in range(args.n_steps):
            obs_buf[t] = obs
            logits, value, _ = agent.forward(obs.astype(np.float32))
            probs = softmax(logits)
            a = np.array([rng.choice(n_actions, p=probs[i]) for i in range(args.n_envs)])
            logp = np.log(probs[np.arange(args.n_envs), a] + 1e-8)
            act_buf[t] = a
            val_buf[t] = value
            logp_buf[t] = logp

            next_obs, r, term, trunc, _ = envs.step(a)
            rew_buf[t] = r
            done_buf[t] = np.logical_or(term, trunc).astype(np.float32)
            cur_ret += r
            env_steps += args.n_envs

            for i, d in enumerate(np.logical_or(term, trunc)):
                if d:
                    ep_returns[i].append(float(cur_ret[i]))
                    cur_ret[i] = 0.0
            obs = next_obs

        # Bootstrap V at end
        _, last_v, _ = agent.forward(obs.astype(np.float32))

        # GAE advantage + returns
        adv_buf = np.zeros_like(rew_buf)
        gae = np.zeros(args.n_envs, dtype=np.float32)
        next_v = last_v
        for t in reversed(range(args.n_steps)):
            mask = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_v * mask - val_buf[t]
            gae = delta + args.gamma * args.gae_lambda * mask * gae
            adv_buf[t] = gae
            next_v = val_buf[t]
        ret_buf = adv_buf + val_buf

        # Advantage normalize
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        # Flatten and update
        B = args.n_steps * args.n_envs
        obs_flat = obs_buf.reshape(B, obs_dim)
        act_flat = act_buf.reshape(B)
        adv_flat = adv_buf.reshape(B)
        ret_flat = ret_buf.reshape(B)

        logits, value, cache = agent.forward(obs_flat)
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-8)
        # Policy gradient: dL/dlogits = (softmax - one_hot) * adv (per row)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B), act_flat] = 1.0
        dlogits_pg = (probs - one_hot) * adv_flat[:, None]
        # Entropy bonus: dL_ent/dlogits = -ent_coef * (log_probs + 1) * probs (approx)
        # Standard entropy regularization: add -ent_coef * H to loss.
        # H = -sum(p log p). dH/dlogits = -(log p + 1) * (p - mean(p))? Use simple form:
        # d(-ent_coef*H)/dlogits[k] = ent_coef * (log p[k] + H) * p[k] — actually simpler:
        # for softmax with entropy H, dH/dlogit_k = -p_k * (log p_k + H).
        H = -(probs * log_probs).sum(axis=1, keepdims=True)
        dlogits_ent = -args.ent_coef * (-probs * (log_probs + H))
        dlogits = dlogits_pg + dlogits_ent
        # Value gradient: dL/dvalue = vf_coef * (value - return)
        dvalue = args.vf_coef * (value - ret_flat)

        grads = agent.backward(cache, dlogits, dvalue)
        # Grad-norm clip
        total_norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        max_norm = 0.5
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for g in grads.values():
                g *= scale
        agent.adam_step(grads)

        if args.log_every and env_steps > 0 and (upd * args.n_envs * args.n_steps) % args.log_every < args.n_envs * args.n_steps:
            recent = [r for lane in ep_returns for r in lane[-5:]]
            avg_ret = sum(recent) / max(1, len(recent))
            elapsed = time.time() - t0
            print(f"step={env_steps:>7} avg_ret={avg_ret:+7.1f} "
                  f"H={H.mean():.2f} V={value.mean():+5.2f} "
                  f"adv={adv_flat.std():.2f} "
                  f"| {env_steps/max(1e-3, elapsed):.0f} env-steps/s")

    envs.close()
    elapsed = time.time() - t0
    total_eps = sum(len(r) for r in ep_returns)
    if total_eps:
        mean_ret = sum(r for lane in ep_returns for r in lane) / total_eps
        last_ret = sum(r for lane in ep_returns for r in lane[-50:]) / max(1, sum(min(50, len(lane)) for lane in ep_returns))
    else:
        mean_ret = float("nan")
        last_ret = float("nan")
    print(f"\n--- A2C control on {args.env} ---")
    print(f"total env-steps: {env_steps}")
    print(f"episodes: {total_eps}, mean return: {mean_ret:+.2f}, "
          f"last-50 mean: {last_ret:+.2f}")
    print(f"wall: {elapsed:.1f}s, throughput: {env_steps/max(1e-3, elapsed):.0f} env-steps/s")


if __name__ == "__main__":
    raise SystemExit(main())

# Next steps after 2026-04-18

As of today's commits, Phase G is feature-complete (v1 cosine bonus,
v2 value-bypass + option-indexed policy, v3 entropy-recovery, v4
learned termination feature-flagged, v5 L1 credit assigner) and M5
(1M-step stability) passes on GridWorld. LunarLander plateaus at
~5% cumulative soft-landing rate with a peak-window of 9.5%, short of
the design-doc `2×` baseline bar.

This document lays out what could come next, organized by cost vs
payoff. Tier 1 and Tier 2 are the current track; Tier 3 and Tier 4
are parked for a future session.

## Tier 1 — cheap experiments (< 30 min each, active now)

Target the LunarLander plateau by touching levers we haven't. The
mechanism hypothesis: kindle's per-step advantage is state-dependent,
not action-sequence-dependent, so precise multi-step landing can't
be rewarded end-to-end. Three untried levers directly address this:

1. **`history_len=64` on L0 v3** — widens the causal-attention
   credit window from 16 to 64 env steps. `H_eff` can then attribute
   end-of-episode reward to a decelerate action ~50 steps back,
   covering most of a typical descent.

2. **`option_horizon=50` on L1 v3** — lets one option commit through
   the whole descent instead of resampling mid-landing every 10 steps.
   L1's goal-latent has time to actually shape the trajectory.

3. **v4 shaping with a positive leg-contact reward** — every current
   homeo term is a negative penalty. Adding a positive
   `leg1 + leg2` signal breaks the all-negative landscape and creates
   the asymmetric gradient a commit-to-a-strategy policy needs.

Pass condition: any config breaks the 10% per-window soft-rate
consistently at 100k steps. Fail: all three plateau ~5–8%.

## Tier 2 — profiling infrastructure (moved to mega-plays)

Redirected 2026-04-18: for kindle on small gym envs the CPU wrapper
time is dominated by meganeura's session dispatches, which already
emit per-pass `.pftrace` events via `meganeura::profiler` if you
call `meganeura::profiler::init()` and `::save(path)`. That's enough
signal for the simple-env regime kindle operates in — there is no
rendering track to coordinate and the python-side overhead is <5%.

The real profiling need is on `mega-plays`, which combines a
rendering loop (blade-graphics) with meganeura training in the same
process and wants a single time-aligned `.pftrace` per run. See the
mega-plays tree for that work.

## Tier 3 — architectural investigations

1. **Phase H design doc (L2 hierarchy)**. Write-before-implement.
2. **Continuous goal-latent instead of fixed option table**. *Active.*
3. **Why Taxi prefers L0** over L1+credit — one-page investigation.
4. **~~Replace shared L0 policy~~ with per-option MLP heads.** Done
   2026-04-19 under feature flag `AgentConfig::per_option_heads`
   (default `true`). Each option now has its own
   `[hidden_dim → action_dim]` fc2 head, selected per-lane via an
   `option_onehot`-gated sum — only the active option's head receives
   gradient. Result on LunarLander at 100k steps / 4 lanes / seed 42:
   **5.25% cumulative soft-rate, 9.0% peak window**, statistically
   indistinguishable from the pre-heads baseline (5.38% / 9.5%). The
   plateau is robust to L0 policy capacity expansion — consistent
   with the "credit horizon, not capacity" hypothesis. Kept on by
   default because the extra head capacity is cheap and leaves the
   Taxi option-differentiation result unchanged. #2 (continuous
   goal-latent) is now the next lever to try.
5. **Why GridWorld stays uniform** across 1M steps — bug hunt.

## Tier 4 — the user-excluded track

M6 learnable reward circuit (now unblocked per README milestones).
Deliberately deferred per user direction.

## Decision principle

The Tier 1 outcomes determine whether LunarLander is a tuning
problem or an architectural one, and that answer dictates whether
Tier 3 work is warranted. Tier 2 infrastructure is independent — it
unblocks profiling-driven throughput and instrumentation work
regardless of Tier 1 results.

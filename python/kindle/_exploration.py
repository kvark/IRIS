"""Per-stream persistent random controls; observations and learning keep their cadence."""

import math
import random


EXPLORATION_PROTOCOL = "kindle-vector-v3"
EXPLORATION_KIND = "persistent-uniform-v1"
EXPLORATION_SEED_XOR = 0x4558_504C_4F52_4501


class PersistentExploration:
    """Choose whether each fixed-length block uses the policy or a held random action."""

    def __init__(self, config, streams, action_count):
        if not isinstance(config, dict) or set(config) != {
            "kind", "probability", "hold_actions", "seed"
        }:
            raise ValueError("invalid exploration configuration fields")
        if config["kind"] != EXPLORATION_KIND:
            raise ValueError("unknown exploration kind")
        probability = config["probability"]
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 < probability <= 1):
            raise ValueError("exploration probability must be in (0, 1]")
        for name, value in (("hold_actions", config["hold_actions"]),
                            ("streams", streams), ("action_count", action_count)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**64:
            raise ValueError("exploration seed must be an unsigned 64-bit integer")
        self._config = dict(config)
        self._action_count = action_count
        self._rngs = [random.Random(((config["seed"] + stream) % 2**64) ^ EXPLORATION_SEED_XOR)
                      for stream in range(streams)]
        self._remaining = [0] * streams
        self._held = [None] * streams
        self._counts = [0] * streams

    @property
    def config(self):
        return dict(self._config)

    @property
    def overridden_actions(self):
        return list(self._counts)

    def actions(self):
        for stream, rng in enumerate(self._rngs):
            if self._remaining[stream] == 0:
                self._held[stream] = (rng.randrange(self._action_count)
                    if rng.random() < self._config["probability"] else None)
                self._remaining[stream] = self._config["hold_actions"]
            self._remaining[stream] -= 1
            self._counts[stream] += self._held[stream] is not None
        return list(self._held)

    def reset(self, streams):
        if (any(type(stream) is not int or not 0 <= stream < len(self._rngs) for stream in streams)
                or len(streams) != len(set(streams))):
            raise ValueError("invalid or repeated exploration reset stream")
        for stream in streams:
            self._remaining[stream] = 0
            self._held[stream] = None

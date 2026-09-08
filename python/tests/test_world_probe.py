import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location(
    "world_probe_example", EXAMPLES / "probe_atari_dynamics.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def recorded_rows():
    config = dict(
        action_count=3,
        intrinsic_reward_scale=0.0,
        extrinsic_reward_scale=1.0,
        horizon=333,
        loss_scales={"future_prediction": 0.25},
    )
    header = dict(
        event="run_start",
        protocol="kindle-vector-v1",
        num_envs=1,
        mode="evaluate_sample",
        seed=7,
        environment_seeds=[7],
        environment="ALE/Pong-v5",
        atari_protocol="published",
        action_repeat=4,
        noop_max=0,
        max_episode_frames=100000,
        full_action_space=True,
        sticky_actions=0.0,
        ale_py_version=probe.ale_py.__version__,
        wrapper_sha256="expected",
        native_extension_sha256="expected",
        config=config,
        model_provenance={},
        action_meanings=["NOOP", "RIGHT", "LEFT"],
        starting_environment_step=10,
        starting_learner_step=20,
        restored_checkpoint={
            "metadata_sha256": "expected",
            "tensor_sha256": dict(
                world="expected", behavior="expected", slow_value="expected"
            ),
        },
    )
    transitions = [
        dict(
            event="transition",
            run_step=i + 1,
            vector_tick=i + 1,
            actions=[i],
            rewards=[float(i)],
            stored_rewards=[[float(i), 0.0]],
            terminated=[i == 1],
            truncated=[False],
            executed_action_frames=[4 * (i + 1)],
        )
        for i in range(2)
    ]
    episode = dict(
        event="episode",
        stream=0,
        episode=0,
        episode_length=2,
        run_step=2,
        stream_step=2,
        episode_return=1.0,
        terminated=True,
        truncated=False,
    )
    return [header, *transitions, episode]


def write_rows(path, rows):
    encoded = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(encoded)
    return encoded


def test_recorded_game_pins_prefix_and_ignores_later_games(tmp_path):
    path = tmp_path / "recorded.jsonl"
    prefix = write_rows(path, recorded_rows())
    path.write_bytes(prefix + b'{"event":"unrelated_tail"}\n')
    header, transitions, record = probe.recorded_first_game(path)
    assert header["seed"] == 7
    assert len(transitions) == 2
    assert record["prefix_bytes"] == len(prefix)
    assert record["prefix_sha256"] == hashlib.sha256(prefix).hexdigest()


@pytest.mark.parametrize(
    "mutation", ["learning", "gap", "stream", "return", "boundary", "after_boundary"]
)
def test_rejects_changed_recording_before_gpu(tmp_path, mutation):
    rows = recorded_rows()
    if mutation == "learning":
        rows.insert(1, {"event": "learner"})
    elif mutation == "gap":
        rows[1]["run_step"] = 2
    elif mutation == "stream":
        rows[0]["num_envs"] = 2
    elif mutation == "return":
        rows[-1]["episode_return"] = 2.0
    elif mutation == "boundary":
        rows[-1]["truncated"] = True
    else:
        rows[1]["terminated"] = [True]
    path = tmp_path / "recorded.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        probe.recorded_first_game(path)


def test_incomplete_recording_is_rejected(tmp_path):
    path = tmp_path / "recorded.jsonl"
    encoded = write_rows(path, recorded_rows())
    path.write_bytes(encoded[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        probe.recorded_first_game(path)


@pytest.mark.parametrize(
    "field",
    [
        "actions",
        "rewards",
        "stored_rewards",
        "terminated",
        "truncated",
        "executed_action_frames",
    ],
)
def test_all_actual_transition_fields_are_checked(field):
    row = copy.deepcopy(recorded_rows()[1])
    row[field] = [None]
    with pytest.raises(ValueError, match="transition differs"):
        probe.verify_transition(row, 0, 0.0, False, False, 4)


@pytest.mark.parametrize(
    "left,right", [([], []), ([1], [1, 2]), ([np.nan], [0]), ([0], [np.inf])]
)
def test_feature_errors_reject_invalid_shapes_and_values(left, right):
    with pytest.raises(ValueError):
        probe.feature_mse(left, right)


@pytest.fixture
def replay(tmp_path, monkeypatch):
    rows = recorded_rows()
    events = []

    class Environment:
        action_space = SimpleNamespace(n=3)
        action_meanings = rows[0]["action_meanings"]
        executed_action_frames = 0
        closed = False
        index = 0

        def reset(self, seed):
            assert seed == 7
            return 0, {}

        def step(self, action):
            events.append(("environment", self.index))
            assert action == self.index
            self.index += 1
            self.executed_action_frames += 4
            return self.index, float(self.index == 2), self.index == 2, False, {}

        def close(self):
            self.closed = True

    class Agent:
        config = rows[0]["config"]
        provenance = {}
        gpu_device = {"mock": True}
        environment_step = 10
        learner_step = 20
        seen = 0
        wrong_action = False

        def begin_episode(self, frame):
            assert frame == 0

        @property
        def visual_observation(self):
            return [float(self.seen)]

        def act(self):
            return 99 if self.wrong_action else self.seen

        def observation_prediction(self):
            return self.visual_observation

        def prior_diagnostic_rollout(self, actions):
            events.append(("forecast", self.seen))
            return [0.5] * len(actions), [
                [float(self.seen + i + 1)] for i in range(len(actions))
            ]

        def prior_behavior_rollout(self, actions):
            events.append(("forecast", self.seen))
            return [0.5] * len(actions), [0.9] * len(actions), [0.0] * len(actions)

        def posterior_reward_prediction(self):
            return float(self.seen == 2)

        def observe(self, frame, **transition):
            assert frame == self.seen + 1
            self.seen = frame
            self.environment_step += 1

    environment, agent = Environment(), Agent()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in [
        "metadata.json",
        "world.safetensors",
        "behavior.safetensors",
        "slow_value.safetensors",
    ]:
        (checkpoint / name).write_bytes(b"test")
    log, output, trace = (
        tmp_path / name for name in ["recorded.jsonl", "result.json", "trace.jsonl"]
    )
    write_rows(log, rows)
    monkeypatch.setattr(probe.gym, "make", lambda *a, **kw: environment)
    monkeypatch.setattr(probe, "DreamerAtariPreprocessing", lambda env, **kw: env)
    monkeypatch.setattr(
        probe.kindle, "Agent", SimpleNamespace(restore=lambda *a: agent)
    )
    monkeypatch.setattr(probe, "sha256_file", lambda path: "expected")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "weights",
            str(checkpoint),
            "--recorded-run",
            str(log),
            "--horizon",
            "2",
            "--stride",
            "1",
            "--output",
            str(output),
            "--trace",
            str(trace),
        ],
    )
    return environment, agent, events, output, trace


def test_forecasts_precede_real_targets_without_updates(replay):
    environment, agent, events, output, trace = replay
    probe.main()
    result = json.loads(output.read_text())
    assert result["sampled_actions_match_source"] is True
    assert result["learner_updates"] == 0
    assert result["sample_count_by_horizon"] == [2, 1]
    assert result["prior_observation_mse_by_horizon"] == [0.0, 0.0]
    assert result["persistence_mse_by_horizon"] == [1.0, 4.0]
    assert result["terminal_count_by_horizon"] == [1, 1]
    assert result["reward_calibration_by_horizon"][0]["zero_predictor_mae"] == 0.5
    assert result["reward_calibration_by_horizon"][0]["posterior_mae"] == 0.0
    assert result["reward_calibration_by_horizon"][1]["prior_mae"] == 0.5
    assert "one_step_prior_mae" not in result["reward_calibration_by_horizon"][1]
    assert environment.closed and agent.seen == 2
    assert events == [("forecast", 0)] * 3 + [("environment", 0)] + [
        ("forecast", 1)
    ] * 3 + [("environment", 1)]
    samples = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(samples) == 3
    assert samples[-1]["continuation_target"] == 0.0


def test_policy_divergence_stops_without_forcing_the_recorded_action(replay):
    environment, agent, events, output, trace = replay
    agent.wrong_action = True
    with pytest.raises(ValueError, match="policy diverged"):
        probe.main()
    assert environment.closed and not events
    assert not output.exists()


def test_outputs_are_not_overwritten(replay):
    environment, agent, events, output, trace = replay
    output.write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        probe.main()
    assert output.read_text() == "preserve" and not events

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import kindle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari
import check_atari_adapter as checker


class Environment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self):
        self.total_frames = 0
        self.closed = False
        self.ale = SimpleNamespace(getFrameNumber=lambda: self.total_frames)

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.length = int(self.np_random.integers(5, 14))
        return self.image(0), {}

    def image(self, action):
        value = (int(self.np_random.integers(0, 128)) + 13 * action + self.steps) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)

    def step(self, action):
        self.total_frames += 1
        self.steps += 1
        return (self.image(action), float(self.steps == 3), self.steps >= self.length,
                False, {"frame_number": self.total_frames})

    def close(self):
        self.closed = True


@pytest.fixture
def environments(monkeypatch):
    created = []

    def make(_):
        raw = Environment()
        created.append(raw)
        return atari.DreamerAtariPreprocessing(raw, noop_max=0, max_episode_frames=100_000)

    def forbid_agent(*_, **__):
        raise AssertionError("adapter diagnostics must not construct a Kindle agent")

    monkeypatch.setattr(checker, "make_environment", make)
    monkeypatch.setattr(checker, "rom_identity", lambda _: dict(path="fixture", sha256="fixture"))
    monkeypatch.setattr(kindle, "Agent", forbid_agent)
    monkeypatch.setattr(kindle, "VectorAgent", forbid_agent)
    return created


def test_interleaved_streams_match_fresh_serial_replay(environments, tmp_path):
    path = tmp_path / "adapter.jsonl"
    report = checker.check_game("fixture", 32, 8001, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1] == report and report["status"] == "adapter_replay_verified"
    assert report["collected_actions"] == report["serial_replay_actions"] == 64
    assert report["learner_updates"] == 0 and not report["agent_constructed"]
    assert not report["competence_evaluated"] and "natural_wins" not in report
    assert report["natural_episodes"] > 0 and report["mean_completed_return"] > 0
    assert report["emulator_resets"][0] != report["emulator_resets"][1]
    assert all(sum(counts) == 32 for counts in report["action_counts"])
    assert len(environments) == 4 and all(env.closed for env in environments)


def test_short_prefix_reports_unfinished_tails_not_zero_score(environments, tmp_path):
    report = checker.check_game("fixture", 1, 8001, tmp_path / "adapter.jsonl")
    assert report["completed_episodes"] == 0
    assert report["mean_completed_return"] is None
    assert report["partial_lengths"] == [1, 1]


@pytest.mark.parametrize("field,value", [
    ("frame_sha256", "bad"), ("reward", 99.0), ("action", 2),
    ("terminated", False), ("truncated", True), ("action_frames", 0),
    ("executed_action_frames", 999),
])
def test_corrupted_replay_is_rejected(environments, tmp_path, field, value):
    path = tmp_path / "adapter.jsonl"
    checker.check_game("fixture", 8, 8001, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    history = [copy.deepcopy(row) for row in rows if row["event"] in ("reset", "transition")]
    row = next(row for row in history if row["event"] == "transition" and row["terminated"])
    row[field] = value if row[field] != value else not value
    with pytest.raises(ValueError, match="serial replay mismatch"):
        checker.verify_serial("fixture", rows[0]["environment_seeds"], history)


@pytest.mark.parametrize("frame", [np.zeros((64, 64, 3), dtype=np.float32),
                                   np.zeros((32, 64, 3), dtype=np.uint8), None])
def test_wrong_frame_contract_is_rejected(frame):
    with pytest.raises(ValueError, match="RGB8"):
        checker.frame_hash(frame)


@pytest.mark.parametrize("mutation", ["clock", "reward", "boundary"])
def test_bad_adapter_outputs_are_rejected(monkeypatch, environments, mutation):
    env = checker.make_environment("fixture")
    env.reset(seed=8001)
    original = env.step

    def broken(action):
        frame, reward, terminated, truncated, info = original(action)
        if mutation == "clock":
            info["frame_number"] += 1
        elif mutation == "reward":
            reward = float("nan")
        else:
            terminated = 1
        return frame, reward, terminated, truncated, info

    monkeypatch.setattr(env, "step", broken)
    with pytest.raises(ValueError):
        checker.step_record(env, 0, 1, 0)
    env.close()


def test_existing_log_is_preserved(environments, tmp_path):
    path = tmp_path / "adapter.jsonl"
    path.write_text("user-owned data")
    with pytest.raises(FileExistsError):
        checker.check_game("fixture", 8, 8001, path)
    assert path.read_text() == "user-owned data"
    assert not environments


@pytest.mark.parametrize("args", [["--steps-per-stream", "0"], ["--seed", "-1"],
                                  ["--environments", "ALE/Pong-v5", "ALE/Pong-v5"]])
def test_invalid_protocol_is_rejected_before_output_creation(monkeypatch, environments, tmp_path, args):
    path = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["check_atari_adapter.py", str(path), *args])
    with pytest.raises(SystemExit) as error:
        checker.main()
    assert error.value.code == 2 and not path.exists() and not environments


def test_failed_replay_retains_failed_summary_without_run_end(monkeypatch, environments, tmp_path):
    path = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["check_atari_adapter.py", str(path), "--environments", "fixture", "--steps-per-stream", "8"])

    def fail(*_):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(checker, "verify_serial", fail)
    with pytest.raises(SystemExit) as error:
        checker.main()
    assert error.value.code == 1
    summary = json.loads((path / "summary.json").read_text())
    assert summary[0]["status"] == "failed" and "injected replay failure" in summary[0]["error"]
    assert all(json.loads(line)["event"] != "run_end" for line in (path / "fixture.jsonl").read_text().splitlines())
    assert all(env.closed for env in environments)

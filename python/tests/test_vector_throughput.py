import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import profile_atari_vector


def fixture(tmp_path):
    config = dict(batch_size=16, batch_length=64, replay_context=1, replay_capacity=4096,
                  train_ratio=0, action_count=18)
    rows = [dict(event="run_start", protocol="kindle-vector-v1", num_envs=2, steps=3072,
                 environment_seeds=[0, 1000003], mode="evaluate_sample", config=config,
                 starting_learner_step=0, starting_environment_step=0, action_repeat=4,
                 agent_construction_seconds=10, native_extension_sha256="native-fixture",
                 runner_sha256="runner-fixture", model_provenance={"perception": {"kind": "dinov3"}})]
    for tick in range(1, 1537):
        actions, elapsed = 2 * tick, tick / 4
        frames = [4 * tick, 3 * tick]
        rows.append(dict(event="transition", run_step=actions, vector_tick=tick,
                         actions=[1, 2], rewards=[0, 0], stored_rewards=[[0, 0], [0, 0]],
                         terminated=[False, False], truncated=[False, False],
                         executed_action_frames=frames))
        if tick in (1024, 1536):
            row = dict(event="progress" if tick == 1024 else "run_end", run_step=actions,
                       vector_ticks=tick, environment_step=actions, learner_step=0,
                       replay_len=2 + actions, training_debt=0, executed_action_frames=frames,
                       total_rewards=[0, 0], episode_counts=[0, 0], partial_returns=[0, 0],
                       partial_lengths=[tick, tick], stage_seconds={"observe": elapsed * 0.9},
                       elapsed_seconds=elapsed, unix_time=1000 + elapsed,
                       actions_per_second=actions / elapsed,
                       aggregate_simulated_wall_ratio=sum(frames) / 60 / elapsed,
                       per_stream_simulated_wall_ratio=[n / 60 / elapsed for n in frames])
            if tick == 1536:
                row.update(reason="budget_complete", learner_updates=0, completed_games=0,
                           natural_wins=0, mean_completed_return=None)
            rows.append(row)
    log, trace = tmp_path / "run.jsonl", tmp_path / "gpu.csv"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    samples = ["timestamp, uuid, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"]
    for timestamp in range(990, 1385):
        date = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")
        memory = 16000 if timestamp == 990 else 8000
        samples.append(f"{date}, GPU-fixture, 50 %, {memory} MiB, 100 W\n")
    trace.write_text("".join(samples))
    return log, trace, rows, samples


def test_throughput_uses_validated_actual_frames_and_preserves_frontend(tmp_path):
    log, trace, _, _ = fixture(tmp_path)
    result = profile_atari_vector.summarize(log, trace)
    assert result["accounting_valid"]
    assert result["window_actions"] == 1024 and result["window_actions_per_stream"] == 512
    assert result["window_seconds"] == 128 and result["actions_per_second"] == 8
    assert result["updates"] == 0 and result["mean_update_seconds"] is None
    assert result["executed_action_frames"] == [2048, 1536]
    assert result["aggregate_simulated_wall_ratio"] == 3584 / 60 / 128
    assert result["per_stream_simulated_wall_ratio"] == [2048 / 60 / 128, 1536 / 60 / 128]
    assert result["gpu_samples"] == 128 and result["mean_gpu_activity"] == 50
    assert result["peak_vram_mib"] == 8000 and result["peak_job_vram_mib"] == 16000
    assert result["gpu_uuid"] == "GPU-fixture"
    assert result["model_provenance"]["perception"]["kind"] == "dinov3"
    assert result["log_sha256"] == profile_atari_vector.sha256_file(log)
    assert result["gpu_trace_sha256"] == profile_atari_vector.sha256_file(trace)


@pytest.mark.parametrize("mutation,message", [
    ("missing_transition", "vector/action counter"),
    ("clock_jump", "window clocks"),
    ("stage_rewind", "stage counters"),
    ("missing_samples", "coverage"),
    ("duplicate_timestamp", "timestamps must increase"),
    ("nonfinite_sample", "invalid GPU sample"),
    ("wrong_gpu", "exactly one selected GPU"),
])
def test_throughput_rejects_invalid_ledger_or_measurement(tmp_path, mutation, message):
    log, trace, rows, samples = fixture(tmp_path)
    if mutation == "missing_transition":
        rows.pop(2)
    elif mutation == "clock_jump":
        rows[-1]["unix_time"] += 10
    elif mutation == "stage_rewind":
        rows[-1]["stage_seconds"]["observe"] = 1
    elif mutation == "missing_samples":
        samples = [samples[0], *samples[1::2]]
    elif mutation == "duplicate_timestamp":
        samples[2] = samples[1]
    elif mutation == "nonfinite_sample":
        samples[-1] = samples[-1].replace("50 %", "nan %")
    else:
        samples[-1] = samples[-1].replace("GPU-fixture", "GPU-other")
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace.write_text("".join(samples))
    with pytest.raises(ValueError, match=message):
        profile_atari_vector.summarize(log, trace)


def test_zero_process_exit_does_not_make_invalid_measurements_complete(monkeypatch, tmp_path):
    directory = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["profile_atari_vector.py", "unused", str(directory), "--num-envs", "2"])
    monitor = SimpleNamespace(terminate=lambda: None, wait=lambda **_: None)
    monkeypatch.setattr(profile_atari_vector.subprocess, "Popen", lambda *_, **__: monitor)
    monkeypatch.setattr(profile_atari_vector.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=0))

    def invalid(*_):
        raise ValueError("incomplete GPU coverage")

    monkeypatch.setattr(profile_atari_vector, "summarize", invalid)
    with pytest.raises(SystemExit) as error:
        profile_atari_vector.main()
    assert error.value.code == 1
    results = json.loads((directory / "summary.json").read_text())
    assert results[0]["status"] == "failed" and results[0]["exit_code"] == 0
    assert results[0]["validation_error"] == "incomplete GPU coverage"
    assert "actions_per_second" not in results[0]

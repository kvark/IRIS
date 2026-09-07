"""Serial, fixed-ratio throughput matrix with a per-job 1 Hz GPU trace.

Measures the new vector implementation at N=1/2/4, not independent learners.
Reports the final 1024-action window, excluding construction and replay prefill.
"""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(log, gpu_trace):
    events = [json.loads(line) for line in log.open()]
    header, final = events[0], events[-1]
    if final["event"] != "run_end" or final["reason"] != "budget_complete":
        raise ValueError("incomplete benchmark")
    start = next(e for e in events if e["event"] == "progress" and e["run_step"] == final["run_step"] - 1024)
    elapsed = final["elapsed_seconds"] - start["elapsed_seconds"]
    reports = [e["report"] for e in events if e["event"] == "learner" and e["run_step"] > start["run_step"]]
    ratio = header["config"]["train_ratio"]
    batch_samples = header["config"]["batch_size"] * header["config"]["batch_length"]
    expected_updates = 0 if header["mode"].startswith("evaluate") else 1024 * ratio / batch_samples
    if len(reports) != expected_updates or final["training_debt"] >= 1:
        raise ValueError("benchmark did not sustain its declared train ratio")
    gpu = []
    for row in csv.DictReader(gpu_trace.open()):
        timestamp = datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
        if start["unix_time"] <= timestamp <= final["unix_time"]:
            gpu.append(row)
    if not gpu or len({r[" uuid"] for r in gpu}) != 1:
        raise ValueError("expected samples from exactly one selected GPU")

    def values(key):
        return [float(r[key].split()[0]) for r in gpu]

    return dict(num_envs=header["num_envs"], mode=header["mode"], batch_size=header["config"]["batch_size"],
                window_actions=1024, window_seconds=elapsed, actions_per_second=1024 / elapsed,
                updates=len(reports), updates_per_second=len(reports) / elapsed,
                mean_update_seconds=statistics.mean(r["timing"]["total_seconds"] for r in reports) if reports else None,
                mean_gpu_activity=statistics.mean(values(" utilization.gpu [%]")),
                mean_power_watts=statistics.mean(values(" power.draw [W]")),
                peak_vram_mib=max(values(" memory.used [MiB]")), gpu_samples=len(gpu),
                stage_seconds={k: final["stage_seconds"][k] - start["stage_seconds"][k] for k in final["stage_seconds"]},
                construction_seconds=header["agent_construction_seconds"],
                native_extension_sha256=header["native_extension_sha256"],
                runner_sha256=header["runner_sha256"], config=header["config"],
                model_provenance=header["model_provenance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder_checkpoint")
    parser.add_argument("--encoder", choices=("levjepa", "dinov3"), default="levjepa")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--num-envs", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3072)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gpu", default="GPU-6869e50d-83aa-bec7-6169-adc413f49b32")
    args = parser.parse_args()
    if args.steps < 3072 or args.steps % 512 or any(n <= 0 or 512 % n for n in args.num_envs):
        parser.error("steps must be >=3072 and divisible by 512; env counts must divide 512")
    if args.batch_size <= 0 or len(set(args.num_envs)) != len(args.num_envs):
        parser.error("batch size must be positive and environment counts distinct")
    # Default T64/context1; each stream starts with one non-action reset frame.
    # Leave the whole final 1024-action window after replay becomes eligible.
    warmup_actions = args.batch_size * 64 + max(args.num_envs) * 63
    if not args.evaluate and args.steps - 1024 < warmup_actions:
        parser.error(f"increase steps: the measured window must start after {warmup_actions} warmup actions")
    args.directory.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("atari_vector.py")
    results = []
    for count in args.num_envs:
        path = args.directory / f"n{count}"
        log, trace = path.with_suffix(".jsonl"), path.with_suffix(".gpu.csv")
        command = [sys.executable, str(runner), args.encoder_checkpoint,
                   "--encoder", args.encoder,
                   "--num-envs", str(count), "--steps", str(args.steps),
                   "--batch-size", str(args.batch_size), "--output", str(log), "--report-every", "512"]
        if args.evaluate:
            command += ["--evaluate", "--train-ratio", "0"]
        print("Starting", " ".join(command), flush=True)
        with trace.open("x") as gpu_output, path.with_suffix(".log").open("x") as run_output:
            monitor = subprocess.Popen(["nvidia-smi", "-i", args.gpu,
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "1"],
                stdout=gpu_output, env={**os.environ, "TZ": "UTC"})
            try:
                process = subprocess.run(command, stdout=run_output, stderr=subprocess.STDOUT, check=False)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        result = (dict(status="failed", num_envs=count, batch_size=args.batch_size,
                       exit_code=process.returncode, command=command, log=str(path.with_suffix(".log")))
                  if process.returncode else dict(status="complete", **summarize(log, trace)))
        results.append(result)
        with path.with_suffix(".summary.json").open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in result.items() if k not in ("config", "stage_seconds", "model_provenance")}), flush=True)
    with (args.directory / "summary.json").open("x") as output:
        json.dump(results, output, indent=2, allow_nan=False)
    if any(result["status"] != "complete" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

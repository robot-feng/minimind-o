#!/usr/bin/env python3
"""Wait for GPU jobs by pidfd, then train and evaluate the single-image TIPS path."""

import ctypes
import json
import math
import os
import re
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/tzq/minimind-o")
sys.path.insert(0, str(ROOT))
from trainer.training_health import LossStabilityTracker


PYTHON = Path("/data/miniconda3/envs/minimind/bin/python")
OUT = ROOT / "out"
RUNS = OUT / "eval_intermediate" / "tips_single_image_20260927"
LOG = OUT / "tips_single_image_train.log"
STATUS = OUT / "tips_single_image_train_status.json"
MODEL_TAG = "sft_i2t_tips_single_20260927"
DATA_PATH = ROOT / "dataset/sft_i2t_mini.parquet"
LOSS_LINE = re.compile(r"Epoch:\[[^]]+\]\((\d+)/(\d+)\), loss: ([^,]+)")
LOG_INTERVAL = 100
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long


def write_status(state, **extra):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def log(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def gpu_compute_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return sorted({int(value.strip()) for value in result.stdout.splitlines() if value.strip()})


def open_pidfd(pid):
    opener = getattr(os, "pidfd_open", None)
    if opener is not None:
        return opener(pid)
    fd = LIBC.syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def wait_for_gpu_release():
    while True:
        pids = gpu_compute_pids()
        if not pids:
            time.sleep(15)
            if not gpu_compute_pids():
                log("All visible GPUs stayed idle for 15 seconds.")
                return
            continue

        poller = select.poll()
        pending = {}
        for pid in pids:
            try:
                fd = open_pidfd(pid)
            except (AttributeError, FileNotFoundError, ProcessLookupError):
                continue
            poller.register(fd, select.POLLIN)
            pending[fd] = pid
        if not pending:
            continue

        log(f"Waiting for GPU compute processes by pidfd: {','.join(map(str, pids))}.")
        while pending:
            for fd, _ in poller.poll():
                poller.unregister(fd)
                pid = pending.pop(fd)
                os.close(fd)
                log(f"GPU process {pid} exited.")


def run_training_stage(name, weight_from, weight_to, mode, accumulation, rate):
    command = [
        str(PYTHON), "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", "4", "--master_port", "29562", "train_sft_omni.py",
        "--vision_only", "--mode", mode,
        "--data_path", "../dataset/sft_i2t_mini.parquet",
        "--from_weight", weight_from, "--save_weight", weight_to,
        "--save_dir", "../out", "--vision_dir", "google/tipsv2-b14",
        "--epochs", "1", "--batch_size", "2",
        "--accumulation_steps", str(accumulation), "--learning_rate", rate,
        "--max_seq_len", "768", "--log_interval", str(LOG_INTERVAL),
        "--use_compile", "0",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "4"})
    write_status("training", stage=name, model_tag=MODEL_TAG, dataset=DATA_PATH.name)
    log(f"Starting {name}: {weight_from} -> {weight_to}.")
    stability = LossStabilityTracker(log_interval=LOG_INTERVAL)
    stable_announced = False
    with LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT / "trainer", env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            match = LOSS_LINE.search(line)
            if not match:
                continue
            step, total = int(match.group(1)), int(match.group(2))
            loss = float(match.group(3))
            if not math.isfinite(loss):
                process.terminate()
                raise RuntimeError(f"Non-finite loss in {name} at step {step}: {loss}")
            if stability.observe(step, total, loss) and not stable_announced:
                stable_announced = True
                write_status(
                    "training_stable", stage=name, stable_observations=stability.observations,
                    last_step=step, total_steps=total, loss=loss, model_tag=MODEL_TAG,
                )
                log(
                    f"MINIMIND_TRAINING_STABLE stage={name} "
                    f"observations={stability.observations} step={step}/{total} loss={loss:.6f}"
                )
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    if not stable_announced:
        raise RuntimeError(f"{name} ended before five consecutive finite loss steps")
    log(f"Training stage {name} completed.")


def run_logged(name, command):
    log(f"Starting {name}.")
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {name} ===\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    log(f"Completed {name}.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if not PYTHON.is_file() or not DATA_PATH.is_file():
        raise FileNotFoundError("The minimind environment or single-image mini I2T dataset is missing")
    if not (OUT / "sft_full_a2a_768.pth").is_file():
        raise FileNotFoundError("Missing base checkpoint out/sft_full_a2a_768.pth")
    for suffix in ("_proj_768.pth", "_768.pth"):
        if (OUT / f"{MODEL_TAG}{suffix}").exists():
            raise FileExistsError(f"Refusing to overwrite {OUT / f'{MODEL_TAG}{suffix}'}")

    write_status("waiting_for_gpu_release", dataset=DATA_PATH.name, model_tag=MODEL_TAG)
    wait_for_gpu_release()
    run_training_stage("vision_projector", "sft_full_a2a", f"{MODEL_TAG}_proj", "vision_proj", 2, "5e-5")
    run_training_stage("single_image_sft", f"{MODEL_TAG}_proj", MODEL_TAG, "all", 4, "5e-6")

    for language, prompt_language in (("zh", "1"), ("en", "0")):
        baseline = RUNS / f"sft_full_a2a_{language}.jsonl"
        trained = RUNS / f"{MODEL_TAG}_{language}.jsonl"
        common = [
            "--text_only", "--mode", "4", "--prompt_lang", prompt_language,
            "--image_dir", "./dataset/eval_omni", "--max_new_tokens", "80",
            "--temperature", "0", "--seed", "42",
        ]
        run_logged(f"{language} baseline on dataset/eval_omni", [
            str(PYTHON), "eval_omni.py", "--weight", "sft_full_a2a", *common,
            "--results_jsonl", str(baseline),
        ])
        run_logged(f"{language} single-image TIPS evaluation", [
            str(PYTHON), "eval_omni.py", "--weight", MODEL_TAG, *common,
            "--results_jsonl", str(trained),
        ])
        comparison = RUNS / f"{MODEL_TAG}_{language}_comparison.json"
        with comparison.open("w", encoding="utf-8") as stream:
            subprocess.run([
                str(PYTHON), "eval_visual_metrics.py", str(baseline), "--compare", str(trained),
            ], cwd=ROOT, check=True, stdout=stream)
        run_logged(f"{language} evaluation chart and table", [
            str(PYTHON), "scripts/plot_visual_comparison.py", str(comparison),
            "--output", str(RUNS / f"{MODEL_TAG}_{language}_comparison.png"),
            "--markdown-output", str(RUNS / f"{MODEL_TAG}_{language}_comparison.md"),
            "--before-label", "sft_full_a2a", "--after-label", MODEL_TAG,
            "--title", f"MiniMind-O TIPS single-image evaluation ({language})",
        ])

    write_status("training_and_evaluation_complete", model_tag=MODEL_TAG, results=str(RUNS))
    log(f"Training and fixed-image evaluation complete: {RUNS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FAILED: {error!r}")
        write_status("failed", error=repr(error), model_tag=MODEL_TAG)
        raise

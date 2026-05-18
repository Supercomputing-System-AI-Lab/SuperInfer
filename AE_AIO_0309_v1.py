#!/usr/bin/env python3
"""SuperInfer artifact-evaluation runner.

Replays the SuperInfer half of ``AE_AIO_0309_v1.py`` (the original AE
script also covered LightLLM / NEO / vLLM-LTR baselines, which lived in
separate conda envs and are out of scope for this repository).

For each ``(model, dataset, rps)`` combination, this script:

1. Spawns the SuperInfer benchmark server
   (:mod:`benchmarks.superinfer.api_server`),
2. Waits for ``"Application startup complete"`` on its stdout,
3. Runs the load generator
   (:mod:`benchmarks.superinfer.api_client`),
4. Tears down both processes.

All SuperInfer-specific knobs are now passed as CLI flags
(see ``--proactive-swap-budget`` etc.) instead of environment variables.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
SERVER_PY = REPO_ROOT / "benchmarks" / "superinfer" / "api_server.py"
CLIENT_PY = REPO_ROOT / "benchmarks" / "superinfer" / "api_client.py"
# Use the venv's Python directly so we don't need to ``activate`` the env
# before each subprocess. See README.md for how to create the venv.
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# (model, dataset, rps_list)  — must match what the SuperInfer paper plots.
EXPERIMENTS: List[Tuple[str, str, List[int]]] = [
    ("Qwen/Qwen2.5-32B-Instruct",       "sharegpt", [5, 10, 13, 15, 18, 20]),
    ("mistralai/Mixtral-8x7B-v0.1",     "sharegpt", [5, 10, 13, 15, 18, 20]),
    ("Qwen/Qwen2.5-32B-Instruct",       "lmsys",    [10, 20, 25, 30, 35, 40]),
    ("mistralai/Mixtral-8x7B-v0.1",     "lmsys",    [10, 20, 25, 30, 35, 40]),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "sharegpt",
                                                    [35, 40, 45, 50, 55, 60]),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "lmsys",
                                                    [50, 60, 68, 75, 88, 100]),
]

PORT = 7777
SWAP_SPACE_GIB = 400
PROACTIVE_SWAP_BUDGET = 2400
NUM_REQ = 20_000
TOTAL_TIME = 120


def server_cmd(model: str) -> str:
    """Bash snippet that launches the SuperInfer api_server."""
    return f"""\
ulimit -n 500000
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export C_INCLUDE_PATH=$CUDA_HOME/include
numactl --cpunodebind=0 --membind=0 {VENV_PYTHON} {SERVER_PY} \\
    --model {model} \\
    --max-num-seqs 16384 --max-num-batched-tokens 16384 \\
    --trust-remote-code --swap-space {SWAP_SPACE_GIB} \\
    --collect-detailed-traces all \\
    --port {PORT} \\
    --proactive-swap-budget {PROACTIVE_SWAP_BUDGET} \\
    --swapper-block-first \\
    --pin-memory-fix \\
    --prefix-cache-fix
"""


def client_cmd(model: str, dataset: str, rps: int, output_dir: str) -> str:
    return f"""\
ulimit -n 500000
{VENV_PYTHON} {CLIENT_PY} \\
    --dataset {dataset} \\
    --num_req {NUM_REQ} --total_time {TOTAL_TIME} \\
    --tokenizer {model} \\
    --burstiness 1 --rps {rps} \\
    --port {PORT} \\
    --output_dir {output_dir}
"""


def _spawn(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def run_one(server: str, client: str) -> None:
    server_proc = _spawn(server)
    print("Server launched.", flush=True)

    server_ready = threading.Event()

    def consume_server_output() -> None:
        for line in server_proc.stdout:  # type: ignore[union-attr]
            if not server_ready.is_set():
                print(line, end="", flush=True)
            if ("Application startup complete" in line
                    and not server_ready.is_set()):
                print("Server ready.", flush=True)
                server_ready.set()

    threading.Thread(target=consume_server_output, daemon=True).start()
    server_ready.wait()

    client_proc = _spawn(client)
    print("Client launched.", flush=True)
    client_finished = False
    for line in client_proc.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        if "Everything Finished." in line:
            client_finished = True
            break

    _kill_group(client_proc)
    _kill_group(server_proc)
    print("Server and Client closed.", flush=True)
    time.sleep(10)

    if not client_finished:
        raise RuntimeError("Client did not finish; see logs above.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "ae_results"),
        help="Where to drop per-run statistics (default: %(default)s).",
    )
    args = parser.parse_args()

    if not VENV_PYTHON.exists():
        raise SystemExit(
            f"Expected venv interpreter at {VENV_PYTHON} but it doesn't "
            f"exist. Create it with ``uv venv --python 3.12 .venv`` and "
            f"install SuperInfer (see README.md).")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    runs = [(model, dataset, rps) for model, dataset, rps_list in EXPERIMENTS
            for rps in rps_list]
    print(f"Will run {len(runs)} (model, dataset, rps) combinations into "
          f"{output_dir}", flush=True)

    for i, (model, dataset, rps) in enumerate(runs, 1):
        print(f"\n=== [{i}/{len(runs)}] {model} | {dataset} | rps={rps} ===",
              flush=True)
        run_one(
            server_cmd(model),
            client_cmd(model, dataset, rps, output_dir),
        )

    print("\nAll experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

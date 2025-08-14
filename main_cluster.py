#!/usr/bin/env python3
"""
main_cluster.py — Cluster-aware CLI that leaves main.py untouched.

New convenience:
  --interactive-app  : one-command workflow that allocates a job, starts 'ollama serve',
                       optionally pulls a model, runs 'streamlit run <app>' and prints the SSH tunnel line.
"""
import argparse
import shlex
import subprocess
import sys
from typing import List, Optional, Dict

from hpc import (
    load_config,
    open_interactive_shell,
    remote_run,
    open_interactive_app,
    submit_app_job,
)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run locally or on an HPC cluster (Slurm).", allow_abbrev=False)
    p.add_argument("--cluster", help="Cluster profile name from clusters.yml. Omit for local run.", default=None)
    p.add_argument("--interactive", action="store_true", help="Open an interactive compute shell via Slurm.")
    p.add_argument("--interactive-app", action="store_true",
                   help="Allocate a job, start ollama (tmux/nohup), optionally pull a model, and run Streamlit.")
    p.add_argument("--submit-app", action="store_true",
                   help="Submit a non-interactive Slurm job that launches ollama and Streamlit.")
    p.add_argument("--partition", help="Slurm partition override (e.g., rosa.p)")
    p.add_argument("--ntasks", type=int, help="Slurm --ntasks override")
    p.add_argument("--cpus-per-task", type=int, help="Slurm --cpus-per-task override")
    p.add_argument("--time", help="Slurm --time override (e.g., 01:00:00)")
    p.add_argument("--account", help="Slurm --account override")
    p.add_argument("--mem", help="Slurm memory (e.g., 8G)")
    p.add_argument("--gpus", help="Slurm GPUs (e.g., gpu:1)")
    p.add_argument("--port", type=int, default=8501,
                   help="Port for Streamlit when using --interactive-app/--submit-app (default: 8501)")
    p.add_argument("--model", type=str, default="", help="Optional Ollama model to ensure present (e.g., llama3.1:8b)")
    p.add_argument("--app", type=str, default="main.py", help="Python file for Streamlit (default: main.py)")
    p.add_argument("--workdir", type=str, default="", help="Remote working directory to cd into before launching app")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run (use '--' to separate).")
    return p.parse_args()

def _collect_overrides(ns: argparse.Namespace) -> Dict[str, Optional[str]]:
    return {
        "partition": ns.partition,
        "ntasks": ns.ntasks,
        "cpus_per_task": ns.cpus_per_task,
        "time": ns.time,
        "account": ns.account,
        "mem": ns.mem,
        "gpus": ns.gpus,
    }

def _stream_local(command: List[str]) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No command provided. Use --interactive/--interactive-app with --cluster, or pass a local command after --", file=sys.stderr)
        return 2
    try:
        proc = subprocess.Popen(command)
        return proc.wait()
    except FileNotFoundError:
        print(f"Executable not found: {command[0]}", file=sys.stderr)
        return 127
    except Exception as e:
        print(f"Error running local command: {e}", file=sys.stderr)
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args()

    if not args.cluster:
        return _stream_local(args.command)

    try:
        cfg = load_config(args.cluster)
    except Exception as e:
        print(f"Error loading cluster '{args.cluster}': {e}", file=sys.stderr)
        return 1

    overrides = _collect_overrides(args)

    # Prefer CLI --workdir, else config 'workdir', else None
    workdir = args.workdir or cfg.get("workdir") or None

    if args.interactive_app and args.submit_app:
        print("Choose only one of --interactive-app or --submit-app", file=sys.stderr)
        return 2

    if args.interactive_app:
        return open_interactive_app(cfg, overrides=overrides, port=args.port, model=(args.model or None), app_py=args.app, workdir=workdir)

    if args.submit_app:
        return submit_app_job(cfg, overrides=overrides, port=args.port, model=(args.model or None), app_py=args.app, workdir=workdir)

    if args.interactive:
        return open_interactive_shell(cfg, overrides=overrides)

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        print("No command specified for remote execution. Use '--' before the command.", file=sys.stderr)
        return 2
    remote_cmd = " ".join(shlex.quote(x) for x in args.command)
    return remote_run(cfg, remote_cmd, overrides=overrides, workdir=workdir)

if __name__ == "__main__":
    raise SystemExit(main())

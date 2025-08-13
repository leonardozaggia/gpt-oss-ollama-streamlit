import argparse
import shlex
import subprocess
import sys
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run commands locally or on an HPC cluster via Slurm",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        help="Cluster profile name from clusters.yml",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive Slurm shell",
    )
    parser.add_argument("--partition", type=str, help="Slurm partition")
    parser.add_argument("--ntasks", type=int, help="Number of tasks")
    parser.add_argument(
        "--cpus-per-task", dest="cpus_per_task", type=int, help="CPUs per task"
    )
    parser.add_argument("--time", type=str, help="Time limit")
    parser.add_argument("--account", type=str, help="Slurm account")
    parser.add_argument("--mem", type=str, help="Memory request")
    parser.add_argument("--gpus", type=str, help="GPU request")
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run; prefix with -- to separate",
    )
    return parser.parse_args()


def _effective_command(cmd: List[str]) -> List[str]:
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd


def main() -> None:
    args = _parse_args()
    cmd = _effective_command(args.cmd)

    if not args.cluster:
        if not cmd:
            print("Error: no command provided for local execution", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    from hpc import load_config, open_interactive_shell, remote_run

    try:
        cfg = load_config(args.cluster)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    cfg["partition"] = args.partition or cfg.get("default_partition")
    cfg["ntasks"] = args.ntasks or cfg.get("default_ntasks")
    cfg["cpus_per_task"] = args.cpus_per_task or cfg.get("default_cpus_per_task")
    cfg["time"] = args.time or cfg.get("default_time")
    cfg["account"] = args.account or cfg.get("account")
    cfg["mem"] = args.mem or cfg.get("mem")
    cfg["gpus"] = args.gpus or cfg.get("gpus")

    missing = [k for k in ["partition", "ntasks", "cpus_per_task"] if not cfg.get(k)]
    if missing:
        print(
            f"Error: missing required Slurm settings for cluster '{args.cluster}': {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.interactive:
        rc = open_interactive_shell(cfg)
        if rc != 0:
            print(f"Remote interactive shell exited with code {rc}", file=sys.stderr)
        sys.exit(rc)

    if not cmd:
        print("Error: no command specified for remote execution", file=sys.stderr)
        sys.exit(1)

    command_str = " ".join(shlex.quote(c) for c in cmd)
    rc = remote_run(cfg, command_str)
    if rc != 0:
        print(f"Remote command failed with exit code {rc}", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
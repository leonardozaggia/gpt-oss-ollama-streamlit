import os
import shlex
import subprocess
from typing import Dict, Any

import yaml


REQUIRED_KEYS = ["host", "user"]


def load_config(name: str, path: str = "clusters.yml") -> Dict[str, Any]:
    """Load cluster configuration from ``clusters.yml``.

    Parameters
    ----------
    name: str
        Name of the cluster profile.
    path: str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary with ``ssh_key`` expanded.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise RuntimeError(f"Config file '{path}' not found") from exc

    clusters = data.get("clusters") or {}
    if name not in clusters:
        raise RuntimeError(f"Cluster '{name}' not found in {path}")

    cfg = clusters[name]
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required keys for cluster '{name}': {', '.join(missing)}"
        )

    if cfg.get("ssh_key"):
        cfg["ssh_key"] = os.path.expanduser(cfg["ssh_key"])
    return cfg


def ssh_run(host: str, user: str, cmd: str, identity_file: str | None = None, pty: bool = False) -> int:
    """Run a command on a remote host via SSH and stream output."""
    ssh_cmd = ["ssh", "-o", "BatchMode=yes"]
    if identity_file:
        ssh_cmd.extend(["-i", identity_file])
    if pty:
        ssh_cmd.append("-t")
    ssh_cmd.append(f"{user}@{host}")
    ssh_cmd.append(cmd)
    try:
        result = subprocess.run(ssh_cmd)
    except FileNotFoundError as exc:
        raise RuntimeError("ssh command not found") from exc
    return result.returncode


def _build_srun_cmd(cfg: Dict[str, Any], interactive: bool, command: str | None = None) -> str:
    part = shlex.quote(str(cfg["partition"]))
    ntasks = shlex.quote(str(cfg["ntasks"]))
    cpus = shlex.quote(str(cfg["cpus_per_task"]))
    pieces = ["srun"]
    if interactive:
        pieces.append("--pty")
    pieces.extend(["-p", part, "--ntasks", ntasks, "--cpus-per-task", cpus])
    if cfg.get("account"):
        pieces.extend(["--account", shlex.quote(str(cfg["account"]))])
    if cfg.get("time"):
        pieces.extend(["--time", shlex.quote(str(cfg["time"]))])
    if cfg.get("mem"):
        pieces.extend(["--mem", shlex.quote(str(cfg["mem"]))])
    if cfg.get("gpus"):
        pieces.extend(["--gpus", shlex.quote(str(cfg["gpus"]))])
    if command:
        pieces.append(command)
    else:
        pieces.append("bash")
    return " ".join(pieces)


def open_interactive_shell(cfg: Dict[str, Any]) -> int:
    cmd = _build_srun_cmd(cfg, True)
    pre = cfg.get("pre_commands") or []
    if pre:
        cmd = " && ".join(pre + [cmd])
    wrapped = f"bash -lc {shlex.quote(cmd)}"
    return ssh_run(
        cfg["host"],
        cfg["user"],
        wrapped,
        identity_file=cfg.get("ssh_key"),
        pty=True,
    )


def remote_run(cfg: Dict[str, Any], command: str) -> int:
    cmd = _build_srun_cmd(cfg, False, command)
    pre = cfg.get("pre_commands") or []
    if pre:
        cmd = " && ".join(pre + [cmd])
    wrapped = f"bash -lc {shlex.quote(cmd)}"
    return ssh_run(
        cfg["host"],
        cfg["user"],
        wrapped,
        identity_file=cfg.get("ssh_key"),
        pty=False,
    )
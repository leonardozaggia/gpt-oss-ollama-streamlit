"""
hpc.py — SSH + Slurm helpers for main_cluster.py
"""
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

REQUIRED_KEYS = ["host", "user", "default_partition", "default_ntasks", "default_cpus_per_task"]

def _expand(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(s)))

def _clusters_yaml_path():
    env = os.environ.get("CLUSTERS_YAML")
    if env:
        return Path(_expand(env)).resolve()
    cwd = Path.cwd() / "clusters.yml"
    if cwd.exists():
        return cwd.resolve()
    return Path(__file__).resolve().with_name("clusters.yml")

def _validate_cfg(name: str, cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg or cfg[k] in ("", None)]
    if missing:
        raise ValueError(f"Cluster '{name}' missing required keys: {', '.join(missing)}")
    for k in ("default_ntasks", "default_cpus_per_task"):
        try:
            cfg[k] = int(cfg[k])
        except Exception:
            raise ValueError(f"Cluster '{name}' key '{k}' must be an integer.")
    if "default_time" in cfg and cfg["default_time"] is None:
        cfg["default_time"] = ""

def load_config(name: str) -> Dict[str, Any]:
    path = _clusters_yaml_path()
    if not path.exists():
        raise FileNotFoundError(f"clusters.yml not found at {path}. Set CLUSTERS_YAML or add clusters.yml to the repo root.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "clusters" not in data or name not in data["clusters"]:
        raise KeyError(f"Cluster '{name}' not found in {path}.")
    cfg = data["clusters"][name] or {}
    for k, v in list(cfg.items()):
        if isinstance(v, str):
            cfg[k] = _expand(v)
        elif isinstance(v, list):
            cfg[k] = [_expand(x) for x in v]
    _validate_cfg(name, cfg)
    return cfg

def _build_ssh_base(host: str, user: str, identity_file: Optional[str] = None, pty: bool = False) -> List[str]:
    cmd = ["ssh"]
    key_exists = bool(identity_file) and Path(identity_file).expanduser().exists()
    if key_exists:
        cmd += ["-i", str(Path(identity_file).expanduser()), "-o", "BatchMode=yes"]
    else:
        cmd += ["-o", "BatchMode=no", "-o", "PreferredAuthentications=publickey,keyboard-interactive,password"]
    if pty:
        cmd += ["-tt"]
    cmd.append(f"{user}@{host}")
    return cmd

def _prepend_pre_commands(pre: Optional[List[str]]) -> str:
    if not pre:
        return ""
    return " && ".join(pre) + " && "

def _compose_srun(cfg: Dict[str, Any], overrides: Dict[str, Any], interactive: bool) -> str:
    part = shlex.quote(overrides.get("partition") or cfg["default_partition"])
    ntasks = int(overrides.get("ntasks") or cfg["default_ntasks"])
    cpus = int(overrides.get("cpus_per_task") or cfg["default_cpus_per_task"])
    account = overrides.get("account") or cfg.get("account") or ""
    timel = overrides.get("time") or cfg.get("default_time") or ""
    mem = overrides.get("mem") or cfg.get("mem") or ""
    gpus = overrides.get("gpus") or cfg.get("gpus") or ""

    parts = ["srun"]
    if interactive:
        parts.append("--pty")
    parts += ["-p", part, f"--ntasks={ntasks}", f"--cpus-per-task={cpus}"]
    if account:
        parts += ["--account", shlex.quote(str(account))]
    if timel:
        parts += ["--time", shlex.quote(str(timel))]
    if mem:
        parts += ["--mem", shlex.quote(str(mem))]
    if gpus:
        parts += ["--gpus", shlex.quote(str(gpus))]
    return " ".join(parts)

def _run_streaming(full_cmd: List[str]) -> int:
    proc = subprocess.Popen(full_cmd)
    return proc.wait()

def ssh_run(host: str, user: str, cmd: str, identity_file: Optional[str] = None, pty: bool = False) -> int:
    base = _build_ssh_base(host, user, identity_file, pty=pty)
    remote = f'bash -lc {shlex.quote(cmd)}'
    full = base + [remote]
    return _run_streaming(full)

def open_interactive_shell(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> int:
    overrides = overrides or {}
    srun = _compose_srun(cfg, overrides, interactive=True)
    srun_bash = f"{srun} bash"
    remote_cmd = _prepend_pre_commands(cfg.get('pre_commands')) + srun_bash
    return ssh_run(cfg["host"], cfg["user"], remote_cmd, identity_file=cfg.get("ssh_key"), pty=True)

def remote_run(cfg: Dict[str, Any], command: str, overrides: Optional[Dict[str, Any]] = None, workdir: Optional[str] = None) -> int:
    overrides = overrides or {}
    srun = _compose_srun(cfg, overrides, interactive=False)
    cd = f"cd {shlex.quote(workdir)} && " if workdir else ""
    remote_cmd = cd + _prepend_pre_commands(cfg.get('pre_commands')) + f"{srun} {command}"
    return ssh_run(cfg["host"], cfg["user"], remote_cmd, identity_file=cfg.get("ssh_key"), pty=False)

def open_interactive_app(cfg: Dict[str, Any],
                         overrides: Optional[Dict[str, Any]] = None,
                         port: int = 8501,
                         model: Optional[str] = None,
                         app_py: str = "main.py",
                         workdir: Optional[str] = None) -> int:
    """
    Allocate an interactive job, then on the compute node:
      - print the allocated node name and the exact 'ssh -L' line to run locally
      - cd into workdir (if provided), otherwise remain in $HOME
      - start 'ollama serve' in tmux (or in background if tmux unavailable)
      - (optionally) ensure a model is available: 'ollama pull <model>'
      - export OLLAMA_HOST and launch 'streamlit run <app_py>' on the chosen port
      - drop the user into an interactive shell (so they can ctrl+c or inspect logs)
    """
    overrides = overrides or {}
    srun = _compose_srun(cfg, overrides, interactive=True)
    cd_line = f'cd {shlex.quote(workdir)} || {{ echo "[bootstrap] ERROR: workdir not found: {workdir}"; pwd; ls -la; exit 2; }}; ' if workdir else ""
    before_streamlit = cd_line + _prepend_pre_commands(cfg.get('pre_commands'))

    bootstrap = f"""
set -e
echo "srun allocation successful."
NODE=$(hostname -s || hostname)
echo "Allocated node: $NODE"
echo
echo ">>> On your LOCAL machine, open another terminal and run this tunnel command:"
echo "ssh -L {port}:$NODE:{port} {cfg['user']}@{cfg['host']}"
echo

# Start or ensure Ollama server
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t ollama 2>/dev/null; then
    echo "[bootstrap] tmux session 'ollama' already exists."
  else
    echo "[bootstrap] starting 'ollama serve' in tmux session 'ollama'..."
    tmux new -d -s ollama 'ollama serve'
    sleep 1
  fi
else
  echo "[bootstrap] tmux not found; starting 'ollama serve' in background via nohup..."
  nohup sh -c 'ollama serve' > ~/ollama_server.log 2>&1 &
  sleep 1
fi

export OLLAMA_HOST=http://127.0.0.1:11434

# Ensure model (optional)
MODEL="{model or ''}"
if [ -n "$MODEL" ]; then
  echo "[bootstrap] ensuring model '$MODEL' is available (pull if missing)..."
  if ! ollama show "$MODEL" >/dev/null 2>&1; then
    ollama pull "$MODEL"
  fi
fi

# Move to workdir and start Streamlit
{before_streamlit}
echo "[bootstrap] CWD: $(pwd)"
if [ ! -f {shlex.quote(app_py)} ]; then
  echo "[bootstrap] ERROR: app file not found: {app_py}"
  ls -la
  exit 2
fi

echo "[bootstrap] launching Streamlit app: {app_py} on port {port}"
echo "           (Press Ctrl+C to stop. Your tunnel should point to localhost:{port})"
streamlit run {shlex.quote(app_py)} --server.port {port} --server.address 0.0.0.0 --server.headless true

exec bash
"""
    remote_cmd = _prepend_pre_commands(cfg.get('pre_commands')) + f"{srun} bash -lc {shlex.quote(bootstrap)}"
    return ssh_run(cfg["host"], cfg["user"], remote_cmd, identity_file=cfg.get("ssh_key"), pty=True)

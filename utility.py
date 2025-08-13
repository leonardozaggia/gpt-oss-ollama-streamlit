import time
import json
import atexit
import signal
import tempfile
import subprocess
from typing import Optional
from pathlib import Path

# Persistent location to remember the last active model across crashes/tab closes
_LAST_MODEL_FILE = Path(tempfile.gettempdir()) / "ollama_last_active_model.json"

# Module-level cache of the current active model (best-effort; Streamlit reruns reuse modules)
_ACTIVE_MODEL: Optional[str] = None

def _run_ollama_stop(model: str) -> bool:
    """Try to stop a model via Python client first, then fall back to CLI."""
    if not model:
        return False
    # 1) Python client (if available and supports stop)
    try:
        import ollama  # type: ignore
        # Some client versions expose `ollama.stop()`, others may not.
        try:
            # Newer clients may support `stop(model=...)`
            if hasattr(ollama, "stop"):
                ollama.stop(model=model)  # type: ignore
                return True
        except Exception:
            pass
    except Exception:
        pass

    # 2) CLI fallback (works with the official Ollama binary)
    try:
        subprocess.run(["ollama", "stop", model], check=False, capture_output=True)
        return True
    except Exception:
        return False

def remember_active_model(model: str) -> None:
    """Record active model both in-memory and in a temp file for crash recovery."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model
    try:
        _LAST_MODEL_FILE.write_text(json.dumps({"model": model, "ts": time.time()}), encoding="utf-8")
    except Exception:
        # best effort
        pass

def shutdown_active_model() -> None:
    """Stop the active model and clear the persisted marker."""
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL:
        _run_ollama_stop(_ACTIVE_MODEL)
    # Also stop whatever is persisted (in case state got desynced)
    try:
        if _LAST_MODEL_FILE.exists():
            data = json.loads(_LAST_MODEL_FILE.read_text(encoding="utf-8"))
            persisted = data.get("model")
            if persisted:
                _run_ollama_stop(persisted)
            _LAST_MODEL_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    _ACTIVE_MODEL = None

def register_process_shutdown() -> None:
    """Register atexit and SIGINT/SIGTERM hooks to stop the model when the app/server exits."""
    # Only register once
    if getattr(register_process_shutdown, "_installed", False):
        return
    atexit.register(shutdown_active_model)

    def _sig_handler(signum, frame):
        try:
            shutdown_active_model()
        finally:
            # Re-raise default handler behavior
            original = signal.getsignal(signum)
            if callable(original):
                original(signum, frame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            # Not all platforms allow setting these
            pass

    setattr(register_process_shutdown, "_installed", True)

def cleanup_stale_from_previous_session() -> None:
    """
    If a previous session crashed or the browser tab was closed, stop whatever model
    was recorded last time, to free VRAM/CPU. This runs on each app start.
    """
    try:
        if _LAST_MODEL_FILE.exists():
            data = json.loads(_LAST_MODEL_FILE.read_text(encoding="utf-8"))
            model = data.get("model")
            if model:
                _run_ollama_stop(model)
            _LAST_MODEL_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def stop_previous_if_changed(previous_model: Optional[str], new_model: str) -> None:
    """When switching models in the UI, stop the previous one to free resources."""
    if previous_model and previous_model != new_model:
        _run_ollama_stop(previous_model)

def maybe_stop_if_idle(last_interaction_ts: float, idle_timeout_sec: int, model: Optional[str]) -> bool:
    """
    If idle for longer than idle_timeout_sec, stop the given model.
    Returns True if a stop was issued.
    """
    if not model:
        return False
    if idle_timeout_sec <= 0:
        return False
    now = time.time()
    if (now - float(last_interaction_ts)) >= idle_timeout_sec:
        _run_ollama_stop(model)
        return True
    return False

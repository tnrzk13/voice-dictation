"""Shared daemon lifecycle helpers used by both batch and live modes."""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .system import notify


def setup_daemon_logging(log_path: str) -> None:
    """Configure daemon logging to file and stderr."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
        ],
    )


def cleanup_socket(socket_path: str) -> None:
    """Remove existing socket file if present."""
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass


def _daemon_config_path(socket_path: str) -> str:
    """Path to the JSON file that stores the daemon's current model args."""
    return socket_path + ".json"


def write_daemon_config(socket_path: str, config: Dict[str, Any]) -> None:
    """Write the daemon's model configuration next to its socket."""
    path = _daemon_config_path(socket_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)


def read_daemon_config(socket_path: str) -> Optional[Dict[str, Any]]:
    """Read the daemon's model configuration if it exists and is valid."""
    path = _daemon_config_path(socket_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def cleanup_daemon_config(socket_path: str) -> None:
    """Remove the daemon config file if present."""
    try:
        os.unlink(_daemon_config_path(socket_path))
    except FileNotFoundError:
        pass


def create_daemon_socket(socket_path: str) -> socket.socket:
    """Create, bind, and listen on a Unix domain socket."""
    cleanup_socket(socket_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(1)
    return sock


def wait_for_socket(socket_path: str, timeout: float, poll_interval: float) -> bool:
    """Poll until the daemon is accepting connections or timeout expires."""
    iterations = int(timeout / poll_interval)
    for _ in range(iterations):
        time.sleep(poll_interval)
        if is_daemon_running(socket_path):
            return True
    return False


def start_daemon_process(
    entry_point: str,
    module_path: str,
    socket_path: str,
    timeout: float,
    poll_interval: float,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """Start a daemon subprocess, trying entry point first then module fallback."""
    suffix = extra_args or []
    try:
        subprocess.Popen(
            [entry_point] + suffix,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        subprocess.Popen(
            [sys.executable, "-m", module_path] + suffix,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    return wait_for_socket(socket_path, timeout, poll_interval)


def is_daemon_running(socket_path: str) -> bool:
    """Check if a daemon is running by attempting a socket connection."""
    if not os.path.exists(socket_path):
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(1.0)
        sock.connect(socket_path)
        return True
    except (ConnectionRefusedError, OSError):
        return False
    finally:
        sock.close()


def stop_daemon(
    socket_path: str,
    pkill_patterns: List[str],
    daemon_name: str,
) -> None:
    """Stop a daemon process by pkill patterns and clean up its socket."""
    daemon_stopped = False

    for pattern in pkill_patterns:
        try:
            result = subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                daemon_stopped = True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    socket_existed = os.path.exists(socket_path)
    try:
        if socket_existed:
            os.remove(socket_path)
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not remove socket file: {e}")

    cleanup_daemon_config(socket_path)

    notify("Stopped")

    if daemon_stopped or socket_existed:
        print(f"{daemon_name} daemon stopped")
    else:
        print(f"No {daemon_name.lower()} daemon was running")

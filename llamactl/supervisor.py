"""Lifecycle management for a single llama-server child process.

One model is loaded at a time: switching models stops the running server and
starts a new one. Output is merged into a bounded ring buffer and fanned out to
websocket subscribers; a background poll of /health drives the state machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import signal
import socket
import time
from collections import deque
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from .config import Config, LaunchParams


POSIX = os.name == "posix"


class ServerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"   # process spawned, model not loaded yet
    READY = "ready"         # /health returns ok
    STOPPING = "stopping"
    ERROR = "error"


TIMING_RE = re.compile(
    r"(prompt eval|eval|total) time\s*=\s*([\d.]+) ms\s*/\s*(\d+)\s*(?:tokens|runs)"
    r"(?:.*?([\d.]+) tokens per second)?"
)
# llama.cpp prefixes lines with a timestamp and a single-letter level (I/W/E/D).
LEVEL_RE = re.compile(r"^[\d.]+\s+([IWED])\s")
FATAL_RE = re.compile(r"(terminate called|std::bad_alloc|error loading model|failed to load model)",
                      re.IGNORECASE)


def _iter_process_cmdlines() -> Any:
    """(pid, argv) for every visible process — procfs first, psutil elsewhere."""
    if Path("/proc/self/cmdline").exists():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes().decode(errors="replace")
            except OSError:
                continue
            yield int(entry.name), [p for p in raw.split("\0") if p]
        return
    try:
        import psutil  # type: ignore
    except ImportError:
        return
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            yield proc.info["pid"], proc.info["cmdline"] or []
        except Exception:
            continue


def find_external_servers(exclude_pid: int | None = None) -> list[dict[str, Any]]:
    """llama-server processes this controller did not start (never touched)."""
    out = []
    for pid, parts in _iter_process_cmdlines():
        if pid == exclude_pid:
            continue
        if not parts or "llama-server" not in os.path.basename(parts[0]):
            continue
        port = None
        for i, arg in enumerate(parts):
            if arg == "--port" and i + 1 < len(parts):
                port = parts[i + 1]
        out.append({"pid": pid, "cmdline": " ".join(parts), "port": port,
                    "binary": parts[0]})
    return out


def port_in_use(host: str, port: int) -> bool:
    family = socket.AF_INET
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


class LlamaSupervisor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = ServerState.STOPPED
        self.proc: asyncio.subprocess.Process | None = None
        self.model_path: str = ""
        self.params: LaunchParams | None = None
        self.argv: list[str] = []
        self.started_at: float = 0.0
        self.ready_at: float = 0.0
        self.exit_code: int | None = None
        self.last_error: str = ""
        self.props: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}
        self.logs: deque[dict[str, Any]] = deque(maxlen=cfg.log_buffer_lines)
        self._subscribers: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=5.0)

    # ---------------------------------------------------------------- helpers

    @property
    def bind(self) -> tuple[str, int]:
        """Host/port of the running server, or what the next start would use."""
        params = self.params or self.cfg.defaults
        return params.host, params.port

    @property
    def base_url(self) -> str:
        """Loopback URL the controller itself uses to reach the model server."""
        host, port = self.bind
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::', '') else host}:{port}"

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    def build_argv(self, model_path: str, params: LaunchParams) -> list[str]:
        argv = [
            str(self.cfg.server_bin),
            "-m", model_path,
            "--host", params.host,
            "--port", str(params.port),
            "--metrics",
            "-ngl", str(params.n_gpu_layers),
            "-c", str(params.ctx_size),
            "-np", str(params.parallel),
        ]
        if params.threads:
            argv += ["-t", str(params.threads)]
        if params.batch_size:
            argv += ["-b", str(params.batch_size)]
        if params.flash_attn:
            argv += ["-fa", params.flash_attn]
        if params.mlock:
            argv.append("--mlock")
        if params.no_mmap:
            argv.append("--no-mmap")
        argv.append("--jinja" if params.jinja else "--no-jinja")
        if params.extra_args.strip():
            argv += shlex.split(params.extra_args)
        return argv

    def add_log(self, line: str, stream: str = "server") -> None:
        record = {"ts": time.time(), "stream": stream, "line": line}
        self.logs.append(record)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    # ------------------------------------------------------------- lifecycle

    async def start(self, model_path: str, params: LaunchParams) -> dict[str, Any]:
        async with self._lock:
            if self.state in (ServerState.STARTING, ServerState.READY):
                await self._stop_locked()

            model = Path(model_path).expanduser()
            if not model.is_file():
                raise FileNotFoundError(f"model not found: {model}")
            if not self.cfg.server_bin.is_file():
                raise FileNotFoundError(f"llama-server not found: {self.cfg.server_bin}")

            self.logs.clear()  # before clear_port so its notes survive
            await self.clear_port(params.host, params.port)
            self.argv = self.build_argv(str(model), params)
            self.model_path = str(model)
            self.params = params
            self.exit_code = None
            self.last_error = ""
            self.props = {}
            self.stats = {}
            self.add_log(f"$ {' '.join(shlex.quote(a) for a in self.argv)}", "controller")

            self.proc = await asyncio.create_subprocess_exec(
                *self.argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.cfg.server_bin.parent),
                start_new_session=POSIX,
            )
            self.state = ServerState.STARTING
            self.started_at = time.time()
            self.ready_at = 0.0
            self._tasks = [
                asyncio.create_task(self._pump_output()),
                asyncio.create_task(self._watch_health()),
            ]
            return self.status()

    def find_orphan(self, port: int | None = None) -> dict[str, Any] | None:
        """A llama-server left on the given port by a previous controller run.

        Matched strictly: same binary and same --port. Anything else holding the
        port belongs to somebody else and is never signalled.
        """
        wanted = str(port if port is not None else self.bind[1])
        for proc in find_external_servers(exclude_pid=self.pid):
            if proc["port"] == wanted and Path(proc["binary"]) == self.cfg.server_bin:
                return proc
        return None

    async def clear_port(self, host: str, port: int) -> None:
        if not port_in_use(host, port):
            return
        orphan = self.find_orphan(port)
        if orphan is None:
            raise RuntimeError(
                f"port {port} is already in use by another process — pick a different "
                f"port or stop that process first"
            )
        self.add_log(
            f"port {port} still held by orphaned llama-server (pid {orphan['pid']}) "
            f"from an earlier run — terminating it",
            "controller",
        )
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(orphan["pid"], sig)
            for _ in range(40):
                await asyncio.sleep(0.25)
                if not port_in_use(host, port):
                    return
        raise RuntimeError(f"could not free port {port} (pid {orphan['pid']})")

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            await self._stop_locked()
            return self.status()

    async def _stop_locked(self, timeout: float = 20.0) -> None:
        proc = self.proc
        if proc is None or proc.returncode is not None:
            self.state = ServerState.STOPPED
            self.proc = None
            return
        self.state = ServerState.STOPPING
        self.add_log("stopping llama-server (SIGTERM)", "controller")
        # signal the whole session so nothing is left behind if it forked
        with contextlib.suppress(ProcessLookupError, PermissionError, AttributeError, OSError):
            if POSIX:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.add_log("timed out, sending SIGKILL", "controller")
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                if POSIX:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
        self.exit_code = proc.returncode
        self.proc = None
        self.state = ServerState.STOPPED
        for task in self._tasks:
            task.cancel()
        self._tasks = []

    async def restart(self) -> dict[str, Any]:
        if not self.model_path or self.params is None:
            raise RuntimeError("nothing to restart — no model has been started yet")
        model, params = self.model_path, self.params
        await self.stop()
        return await self.start(model, params)

    async def shutdown(self) -> None:
        await self.stop()
        await self._client.aclose()

    # --------------------------------------------------------------- workers

    async def _pump_output(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip("\n")
                self.add_log(line)
                self._inspect_line(line)
        except asyncio.CancelledError:
            raise
        finally:
            code = await proc.wait()
            if self.state not in (ServerState.STOPPING, ServerState.STOPPED):
                self.exit_code = code
                self.state = ServerState.ERROR if code not in (0, -15) else ServerState.STOPPED
                self.add_log(f"llama-server exited with code {code}", "controller")
                self.proc = None

    def _inspect_line(self, line: str) -> None:
        match = TIMING_RE.search(line)
        if match:
            kind, ms, tokens, tps = match.groups()
            key = {"prompt eval": "prompt", "eval": "generation", "total": "total"}[kind]
            entry = {"ms": float(ms), "tokens": int(tokens)}
            if tps:
                entry["tokens_per_second"] = float(tps)
            self.stats[key] = entry
            self.stats["updated_at"] = time.time()
            return
        level = LEVEL_RE.match(line)
        if (level and level.group(1) == "E") or FATAL_RE.search(line):
            self.last_error = line.strip()[:400]

    async def _watch_health(self) -> None:
        """Poll /health until ready, then keep props and slot info fresh."""
        try:
            while True:
                if self.proc is None or self.proc.returncode is not None:
                    return
                try:
                    response = await self._client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        if self.state is ServerState.STARTING:
                            self.state = ServerState.READY
                            self.ready_at = time.time()
                            self.last_error = ""  # startup warnings are not failures
                            load_time = self.ready_at - self.started_at
                            self.add_log(f"model ready in {load_time:.1f}s", "controller")
                            await self._refresh_props()
                    elif self.state is ServerState.READY:
                        self.state = ServerState.STARTING
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0 if self.state is ServerState.STARTING else 5.0)
        except asyncio.CancelledError:
            raise

    async def _refresh_props(self) -> None:
        try:
            response = await self._client.get(f"{self.base_url}/props")
            if response.status_code == 200:
                self.props = response.json()
        except (httpx.HTTPError, ValueError):
            pass

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "state": self.state.value,
            "pid": self.pid,
            "model_path": self.model_path,
            "model_name": Path(self.model_path).name if self.model_path else "",
            "params": asdict(self.params) if self.params else None,
            "argv": self.argv,
            "url": self.base_url,
            "bind_host": self.bind[0],
            "bind_port": self.bind[1],
            "uptime": round(now - self.started_at, 1) if self.started_at and self.pid else 0,
            "load_seconds": round(self.ready_at - self.started_at, 1) if self.ready_at else None,
            "exit_code": self.exit_code,
            "last_error": self.last_error,
            "stats": self.stats,
            "model_meta": {
                "n_ctx": self.props.get("default_generation_settings", {}).get("n_ctx"),
                "slots": self.props.get("total_slots"),
                "ftype": self.props.get("model_ftype"),
                "build": self.props.get("build_info", ""),
                "chat_template": bool(self.props.get("chat_template")),
                "modalities": self.props.get("modalities", {}),
            }
            if self.props
            else {},
        }

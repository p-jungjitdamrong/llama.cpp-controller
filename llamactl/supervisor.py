"""Lifecycle management for llama-server child processes.

Several models can run at once: each gets its own `ServerInstance` on its own
port, with its own log ring buffer and health poll. `SupervisorPool` owns them,
allocates ports, and fans every log line out to websocket subscribers tagged
with the instance it came from.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

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
from typing import Any, Callable

import httpx

from .config import CONFIG_PATH, Config, LaunchParams

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


def _ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def find_external_servers(exclude_pids: set[int] | None = None) -> list[dict[str, Any]]:
    """llama-server processes this controller did not start (never touched).

    A router spawns a child llama-server per loaded model; those belong to us
    through their parent, so they are excluded too — they are stopped by
    unloading the model, not by killing the process.
    """
    exclude = exclude_pids or set()
    out = []
    for pid, parts in _iter_process_cmdlines():
        if pid in exclude or _ppid(pid) in exclude:
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


def estimate_footprint(model_path: str, params: LaunchParams) -> dict[str, int]:
    """Rough RAM a model will need: weights plus KV cache plus slack.

    The weights are the file size (mmap still needs the pages resident to be
    fast). The KV cache is 2 bytes per element for K and V across every layer,
    which at a large context can rival the model itself.
    """
    from .gguf import read_metadata  # local import keeps module load cheap

    path = Path(model_path)
    try:
        weights = path.stat().st_size
    except OSError:
        weights = 0
    meta = read_metadata(path) if path.is_file() else {}

    layers = meta.get("n_layer") or 32
    embd = meta.get("n_embd") or 4096
    heads = meta.get("n_head") or 32
    heads_kv = meta.get("n_head_kv") or heads
    head_dim = max(1, embd // max(1, heads))
    ctx = max(params.ctx_size or 4096, 512) * max(1, params.parallel)
    kv = 2 * 2 * layers * ctx * heads_kv * head_dim  # K and V, fp16

    overhead = 350 * 1024**2  # compute buffers, runtime, http server
    return {"weights": weights, "kv_cache": kv, "overhead": overhead,
            "total": weights + kv + overhead}


def port_in_use(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


class ServerInstance:
    """One llama-server process serving one model."""

    def __init__(self, instance_id: int, cfg: Config, model_path: str,
                 params: LaunchParams, on_log: Callable[[dict[str, Any]], None]) -> None:
        self.id = instance_id
        self.cfg = cfg
        self.model_path = model_path
        self.params = params
        self._on_log = on_log
        self.state = ServerState.STOPPED
        self.proc: asyncio.subprocess.Process | None = None
        self.argv: list[str] = []
        self.started_at = 0.0
        self.ready_at = 0.0
        self.exit_code: int | None = None
        self.last_error = ""
        self.props: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}
        self.router_models: list[dict[str, Any]] = []
        self.logs: deque[dict[str, Any]] = deque(maxlen=cfg.log_buffer_lines)
        self._tasks: list[asyncio.Task] = []
        self._client = httpx.AsyncClient(timeout=5.0)

    # ---------------------------------------------------------------- basics

    @property
    def is_router(self) -> bool:
        return self.params.mode == "router"

    @property
    def name(self) -> str:
        if self.is_router:
            return f"router: {Path(self.params.models_dir).name or 'models'}"
        return Path(self.model_path).name

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    @property
    def base_url(self) -> str:
        host = self.params.host
        loopback = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        return f"http://{loopback}:{self.params.port}"

    def write_router_preset(self) -> Path:
        """Turn the launch params into an INI the router applies to every child.

        `[*]` is the section llama.cpp applies to all models; keys are long flag
        names without the leading dashes.
        """
        params = self.params
        lines = ["# generated by llama-controller — edit the launch params in the UI",
                 "[*]",
                 f"gpu-layers = {params.n_gpu_layers}",
                 f"ctx-size = {params.ctx_size}",
                 f"parallel = {params.parallel}",
                 f"jinja = {1 if params.jinja else 0}",
                 f"mmap = {0 if params.no_mmap else 1}"]
        if params.threads:
            lines.append(f"threads = {params.threads}")
        if params.batch_size:
            lines.append(f"batch-size = {params.batch_size}")
        if params.flash_attn:
            lines.append(f"flash-attn = {params.flash_attn}")
        if params.mlock:
            lines.append("mlock = 1")
        path = CONFIG_PATH.parent / f"router-preset-{params.port}.ini"
        path.write_text("\n".join(lines) + "\n")
        return path

    def add_log(self, line: str, stream: str = "server") -> None:
        record = {"ts": time.time(), "stream": stream, "line": line,
                  "instance": self.id, "model": self.name}
        self.logs.append(record)
        self._on_log(record)

    def build_argv(self) -> list[str]:
        params = self.params
        argv = [
            str(self.cfg.server_bin),
            "--host", params.host,
            "--port", str(params.port),
            "--metrics",
        ]
        if self.is_router:
            argv += ["--models-dir", str(Path(params.models_dir).expanduser()),
                     "--models-max", str(params.models_max),
                     "--models-autoload" if params.models_autoload else "--no-models-autoload"]
            extra = shlex.split(params.extra_args) if params.extra_args.strip() else []
            # Without a preset the router starts every child with plain defaults —
            # no -ngl, so the model runs on CPU. Hand it our launch params unless
            # the user supplied a preset file of their own.
            if not any(a.startswith("--models-preset") for a in extra):
                argv += ["--models-preset", str(self.write_router_preset())]
            return argv + extra

        argv += [
            "-m", self.model_path,
            "-a", Path(self.model_path).stem,
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

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.argv = self.build_argv()
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

    async def stop(self, timeout: float = 20.0) -> None:
        proc = self.proc
        if proc is None or proc.returncode is not None:
            self.state = ServerState.STOPPED
            self.proc = None
            await self._cancel_tasks()
            return
        self.state = ServerState.STOPPING
        self.add_log("stopping llama-server (SIGTERM)", "controller")
        # signal the whole session so nothing is left behind if it forked
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
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
        await self._cancel_tasks()

    async def _cancel_tasks(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks = []

    async def close(self) -> None:
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
            entry: dict[str, Any] = {"ms": float(ms), "tokens": int(tokens)}
            if tps:
                entry["tokens_per_second"] = float(tps)
            self.stats[key] = entry
            self.stats["updated_at"] = time.time()
            return
        level = LEVEL_RE.match(line)
        if (level and level.group(1) == "E") or FATAL_RE.search(line):
            self.last_error = line.strip()[:400]

    async def _watch_health(self) -> None:
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
                            self.add_log(
                                f"model ready in {self.ready_at - self.started_at:.1f}s",
                                "controller")
                            await self._refresh_props()
                        elif self.is_router:
                            await self.refresh_router_models()
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
        if self.is_router:
            await self.refresh_router_models()

    async def refresh_router_models(self) -> None:
        """What the router knows about, and which of them are resident."""
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            if response.status_code != 200:
                return
            data = response.json().get("data", [])
        except (httpx.HTTPError, ValueError):
            return
        self.router_models = [
            {
                "id": item.get("id", ""),
                "status": (item.get("status") or {}).get("value", "unknown"),
                "source": item.get("source", ""),
                "modalities": (item.get("architecture") or {}).get("input_modalities", []),
            }
            for item in data
        ]

    async def router_action(self, action: str, model_id: str) -> dict[str, Any]:
        """Ask the router to load or unload one of its models."""
        if not self.is_router:
            raise RuntimeError("this instance is not a router")
        if action not in ("load", "unload"):
            raise ValueError(action)
        response = await self._client.post(
            f"{self.base_url}/models/{action}", json={"model": model_id}, timeout=300.0
        )
        self.add_log(f"router {action} '{model_id}' -> HTTP {response.status_code}", "controller")
        await self.refresh_router_models()
        return {"status_code": response.status_code, "models": self.router_models}

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "id": self.id,
            "state": self.state.value,
            "pid": self.pid,
            "mode": self.params.mode,
            "is_router": self.is_router,
            "router_models": self.router_models,
            "model_path": self.model_path,
            "model_name": self.name,
            "params": asdict(self.params),
            "argv": self.argv,
            "url": self.base_url,
            "bind_host": self.params.host,
            "bind_port": self.params.port,
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
            } if self.props else {},
        }


class SupervisorPool:
    """Every llama-server this controller runs, one per model."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.instances: dict[int, ServerInstance] = {}
        self._next_id = 1
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        # controller-level lines that belong to no single instance
        self.system_logs: deque[dict[str, Any]] = deque(maxlen=200)

    def log(self, line: str) -> None:
        record = {"ts": time.time(), "stream": "controller", "line": line,
                  "instance": None, "model": "controller"}
        self.system_logs.append(record)
        self._publish(record)

    # ------------------------------------------------------------ log fanout

    def _publish(self, record: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def recent_logs(self, limit: int = 300, instance_id: int | None = None) -> list[dict[str, Any]]:
        if instance_id is not None:
            instance = self.instances.get(instance_id)
            return list(instance.logs)[-limit:] if instance else []
        merged: list[dict[str, Any]] = list(self.system_logs)
        for instance in self.instances.values():
            merged.extend(instance.logs)
        merged.sort(key=lambda r: r["ts"])
        return merged[-limit:]

    # ---------------------------------------------------------------- lookup

    @property
    def pids(self) -> set[int]:
        return {i.pid for i in self.instances.values() if i.pid}

    def get(self, instance_id: int) -> ServerInstance | None:
        return self.instances.get(instance_id)

    def by_model(self, model_path: str) -> ServerInstance | None:
        for instance in self.instances.values():
            if instance.model_path == model_path and instance.state is not ServerState.STOPPED:
                return instance
        return None

    def ready(self) -> list[ServerInstance]:
        return [i for i in self.instances.values() if i.state is ServerState.READY]

    def status(self) -> list[dict[str, Any]]:
        return [i.status() for i in sorted(self.instances.values(), key=lambda i: i.id)]

    def used_ports(self) -> set[int]:
        return {i.params.port for i in self.instances.values() if i.pid}

    # ------------------------------------------------------------- lifecycle

    def find_orphan(self, port: int) -> dict[str, Any] | None:
        """A llama-server on `port` left behind by an earlier controller run."""
        wanted = str(port)
        for proc in find_external_servers(exclude_pids=self.pids):
            if proc["port"] == wanted and Path(proc["binary"]) == self.cfg.server_bin:
                return proc
        return None

    async def clear_port(self, host: str, port: int) -> bool:
        """Free `port` if one of our own orphans holds it. True if it was cleared."""
        if not port_in_use(host, port):
            return False
        orphan = self.find_orphan(port)
        if orphan is None:
            raise RuntimeError(
                f"port {port} is in use by another process — pick a different port"
            )
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(orphan["pid"], sig)
            for _ in range(40):
                await asyncio.sleep(0.25)
                if not port_in_use(host, port):
                    return True
        raise RuntimeError(f"could not free port {port} (pid {orphan['pid']})")

    async def _allocate_port(self, params: LaunchParams) -> tuple[int, str]:
        """Return a usable port and a note if it differs from the one requested."""
        wanted = params.port
        if wanted not in self.used_ports():
            with contextlib.suppress(RuntimeError):
                if await self.clear_port(params.host, wanted):
                    return wanted, f"cleared an orphaned llama-server from port {wanted}"
            if not port_in_use(params.host, wanted):
                return wanted, ""
        for candidate in range(wanted + 1, wanted + 64):
            if candidate in self.used_ports() or port_in_use(params.host, candidate):
                continue
            return candidate, f"port {wanted} was taken, using {candidate} instead"
        raise RuntimeError(f"no free port found near {wanted}")

    def check_memory(self, model_path: str, params: LaunchParams) -> dict[str, Any]:
        """Compare the estimated footprint against what is genuinely spare.

        Free memory alone is not enough to decide with: a model that is still
        loading has barely touched its pages, so the machine looks emptier than
        it is about to be. Everything already started is therefore counted at its
        full estimated size, and the smaller of the two views wins.

        Getting this wrong is not a polite failure — an over-commit here means
        the kernel OOM killer, and on a box without swap it can take the whole
        machine down.
        """
        from .metrics import read_memory

        memory = read_memory()
        estimate = estimate_footprint(model_path, params)
        committed = sum(
            estimate_footprint(i.model_path, i.params)["total"]
            for i in self.instances.values() if i.pid and i.model_path
        )
        reserve = int(memory.get("total", 0) * 0.10) + 512 * 1024**2  # OS + page cache
        headroom = max(0, memory.get("total", 0) - committed - reserve)
        budget = min(memory.get("available", 0), headroom)

        estimate.update({
            "available": memory.get("available", 0),
            "committed": committed,
            "headroom": headroom,
            "budget": budget,
            "fits": budget >= estimate["total"],
        })
        return estimate

    async def start(self, model_path: str, params: LaunchParams,
                    force: bool = False) -> dict[str, Any]:
        async with self._lock:
            if not self.cfg.server_bin.is_file():
                raise FileNotFoundError(f"llama-server not found: {self.cfg.server_bin}")

            if params.mode == "router":
                directory = Path(params.models_dir).expanduser()
                if not directory.is_dir():
                    raise FileNotFoundError(f"models directory not found: {directory}")
                params.models_dir = str(directory)
                for other in self.instances.values():
                    if (other.is_router and other.pid
                            and other.params.models_dir == params.models_dir):
                        raise RuntimeError(
                            f"a router for {directory} is already running on port "
                            f"{other.params.port}"
                        )
                resolved = ""
            else:
                model = Path(model_path).expanduser()
                if not model.is_file():
                    raise FileNotFoundError(f"model not found: {model}")
                existing = self.by_model(str(model))
                if existing is not None:
                    raise RuntimeError(
                        f"{model.name} is already running on port {existing.params.port} "
                        f"— stop or restart that instance instead"
                    )
                resolved = str(model)

                fit = self.check_memory(resolved, params)
                if not fit["fits"] and not force:
                    gb = 1024**3
                    raise MemoryError(
                        f"{model.name} needs about {fit['total'] / gb:.1f} GB "
                        f"({fit['weights'] / gb:.1f} GB weights + "
                        f"{fit['kv_cache'] / gb:.1f} GB KV cache at {params.ctx_size} "
                        f"context) and only {fit['budget'] / gb:.1f} GB is spare "
                        f"({fit['committed'] / gb:.1f} GB is already promised to "
                        f"running models). Stop one, lower the context, or override."
                    )

            port, note = await self._allocate_port(params)
            params.port = port

            instance = ServerInstance(self._next_id, self.cfg, resolved, params, self._publish)
            self._next_id += 1
            self.instances[instance.id] = instance
            if note:
                instance.add_log(note, "controller")
            await instance.start()
            return instance.status()

    async def stop(self, instance_id: int) -> dict[str, Any]:
        instance = self.instances.get(instance_id)
        if instance is None:
            raise KeyError(instance_id)
        await instance.stop()
        return instance.status()

    async def restart(self, instance_id: int) -> dict[str, Any]:
        instance = self.instances.get(instance_id)
        if instance is None:
            raise KeyError(instance_id)
        await instance.stop()
        await instance.start()
        return instance.status()

    async def remove(self, instance_id: int) -> None:
        instance = self.instances.pop(instance_id, None)
        if instance is not None:
            await instance.close()

    async def autostart(self, entries: list[dict[str, Any]], timeout: float = 300.0) -> None:
        """Bring up saved servers one at a time, waiting for each to settle.

        Sequential on purpose: loading two large models at once thrashes disk and
        can put the box into swap before either is usable.
        """
        if not entries:
            return
        self.log(f"autostart: bringing up {len(entries)} saved server(s)")
        for entry in entries:
            params = LaunchParams.from_dict(entry.get("params") or {})
            label = Path(entry.get("model_path") or params.models_dir or "?").name
            try:
                status = await self.start(entry.get("model_path", ""), params)
            except Exception as exc:
                self.log(f"autostart: {label} failed — {exc}")
                continue
            instance = self.instances[status["id"]]
            deadline = time.monotonic() + timeout
            while instance.state is ServerState.STARTING and time.monotonic() < deadline:
                await asyncio.sleep(1.0)
            self.log(f"autostart: {instance.name} is {instance.state.value}"
                     f" on port {instance.params.port}")

    async def shutdown(self) -> None:
        await asyncio.gather(*(i.close() for i in self.instances.values()),
                             return_exceptions=True)
        self.instances.clear()

"""System sampling.

On Linux everything comes straight from /proc and /sys, so the controller has no
runtime dependencies at all. On other platforms it falls back to psutil if that
happens to be installed, and reports what it can. GPU readings are delegated to
the vendor backends in `gpu.py`.
"""

from __future__ import annotations

import os
import platform
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from .gpu import GpuMonitor

HAS_PROC = Path("/proc/stat").exists()

try:  # optional, only needed off Linux
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read_text(path: Path | str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _read_int(path: Path | str) -> int | None:
    raw = _read_text(path)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


class CpuSampler:
    """Cumulative CPU counters turned into per-interval percentages."""

    def __init__(self) -> None:
        self._prev: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _parse_proc() -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                values = [int(v) for v in parts[1:]]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                out[parts[0]] = (sum(values), idle)
        return out

    def sample(self) -> dict[str, Any]:
        if not HAS_PROC:
            if psutil is None:
                return {"percent": 0.0, "per_core": []}
            per_core = psutil.cpu_percent(percpu=True)
            return {
                "percent": round(sum(per_core) / len(per_core), 1) if per_core else 0.0,
                "per_core": [round(v, 1) for v in per_core],
            }

        current = self._parse_proc()
        total_pct = 0.0
        per_core: list[float] = []
        for key, (total, idle) in current.items():
            prev = self._prev.get(key)
            pct = 0.0
            if prev:
                d_total = total - prev[0]
                d_idle = idle - prev[1]
                if d_total > 0:
                    pct = max(0.0, min(100.0, 100.0 * (d_total - d_idle) / d_total))
            if key == "cpu":
                total_pct = pct
            else:
                per_core.append(round(pct, 1))
        self._prev = current
        return {"percent": round(total_pct, 1), "per_core": per_core}


class ProcessSampler:
    """CPU/RSS of the supervised llama-server process."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, float]] = {}

    def sample(self, pid: int | None) -> dict[str, Any] | None:
        if not pid:
            return None
        return self._sample_proc(pid) if HAS_PROC else self._sample_psutil(pid)

    def retain(self, pids: Iterable[int]) -> None:
        """Forget CPU baselines for processes that are gone."""
        keep = set(pids)
        for pid in list(self._prev):
            if pid not in keep:
                del self._prev[pid]

    def _cpu_percent(self, pid: int, cpu_seconds: float) -> float:
        now = time.monotonic()
        percent = 0.0
        prev = self._prev.get(pid)
        if prev and now > prev[0]:
            percent = max(0.0, 100.0 * (cpu_seconds - prev[1]) / (now - prev[0]))
        self._prev[pid] = (now, cpu_seconds)
        return round(percent, 1)

    def _sample_proc(self, pid: int) -> dict[str, Any] | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            self._prev.pop(pid, None)
            return None

        # comm can contain spaces and parens; fields are safe after the last ')'
        fields = stat[stat.rindex(")") + 2 :].split()
        cpu_seconds = (int(fields[11]) + int(fields[12])) / CLK_TCK
        threads = int(fields[17])
        start_ticks = int(fields[19])

        rss = vms = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmSize:"):
                vms = int(line.split()[1]) * 1024

        uptime = float((_read_text("/proc/uptime") or "0").split()[0])
        return {
            "pid": pid,
            "cpu_percent": self._cpu_percent(pid, cpu_seconds),
            "rss": rss,
            "vms": vms,
            "threads": threads,
            "elapsed": round(uptime - start_ticks / CLK_TCK, 1),
        }

    def _sample_psutil(self, pid: int) -> dict[str, Any] | None:
        if psutil is None:
            return None
        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                times = proc.cpu_times()
                memory = proc.memory_info()
                return {
                    "pid": pid,
                    "cpu_percent": self._cpu_percent(pid, times.user + times.system),
                    "rss": memory.rss,
                    "vms": memory.vms,
                    "threads": proc.num_threads(),
                    "elapsed": round(time.time() - proc.create_time(), 1),
                }
        except Exception:
            self._prev.pop(pid, None)
            return None


def read_memory() -> dict[str, Any]:
    if HAS_PROC:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        swap_total = info.get("SwapTotal", 0)
        swap_used = swap_total - info.get("SwapFree", 0)
        cached = info.get("Cached", 0) + info.get("Buffers", 0)
    elif psutil is not None:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total, available = virtual.total, virtual.available
        swap_total, swap_used = swap.total, swap.used
        cached = getattr(virtual, "cached", 0)
    else:
        return {"total": 0, "available": 0, "used": 0, "cached": 0, "percent": 0.0,
                "swap_total": 0, "swap_used": 0}

    return {
        "total": total,
        "available": available,
        "used": total - available,
        "cached": cached,
        "percent": round(100.0 * (total - available) / total, 1) if total else 0.0,
        "swap_total": swap_total,
        "swap_used": swap_used,
    }


def read_cpu_temp() -> float | None:
    if HAS_PROC:
        for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            if (_read_text(hwmon / "name") or "") not in (
                "k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz"
            ):
                continue
            for label_file in sorted(hwmon.glob("temp*_label")):
                if (_read_text(label_file) or "").lower() in ("tctl", "tdie", "package id 0"):
                    value = _read_int(str(label_file).replace("_label", "_input"))
                    if value is not None:
                        return round(value / 1000, 1)
            value = _read_int(hwmon / "temp1_input")
            if value is not None:
                return round(value / 1000, 1)
        return None
    if psutil is not None and hasattr(psutil, "sensors_temperatures"):
        for entries in (psutil.sensors_temperatures() or {}).values():
            for entry in entries:
                if entry.current:
                    return round(entry.current, 1)
    return None


def read_loadavg() -> list[float]:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        return [0.0, 0.0, 0.0]


def cpu_model_name() -> str:
    if HAS_PROC:
        for line in (_read_text("/proc/cpuinfo") or "").splitlines():
            if line.startswith(("model name", "Model")):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        try:
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, Exception):
            pass
    return platform.processor() or platform.machine()


class Monitor:
    """One sample per tick, kept in a bounded history for the dashboard charts."""

    def __init__(self, history_size: int = 300, disk_paths: Iterable[str] = ("/",)) -> None:
        self.cpu = CpuSampler()
        self.proc = ProcessSampler()
        self.gpu = GpuMonitor()
        self.disk_paths = list(disk_paths)
        self.history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.cpu.sample()  # prime the counters so the first real sample is useful

    def static_info(self) -> dict[str, Any]:
        return {
            "hostname": platform.node(),
            "platform": f"{platform.system()} {platform.release()}",
            "cpu_model": cpu_model_name(),
            "cpu_count": os.cpu_count(),
            "mem_total": read_memory()["total"],
            "gpus": self.gpu.describe(),
            "metrics_source": "procfs" if HAS_PROC else ("psutil" if psutil else "limited"),
        }

    def sample_many(self, pids: Iterable[int]) -> dict[str, Any]:
        """A system sample plus per-process readings for several servers."""
        pids = list(pids)
        self.proc.retain(pids)
        sample = self.sample()
        sample["processes"] = {
            pid: proc for pid in pids if (proc := self.proc.sample(pid)) is not None
        }
        totals = {"cpu_percent": 0.0, "rss": 0, "threads": 0}
        for proc in sample["processes"].values():
            totals["cpu_percent"] = round(totals["cpu_percent"] + proc["cpu_percent"], 1)
            totals["rss"] += proc["rss"]
            totals["threads"] += proc["threads"]
        totals["count"] = len(sample["processes"])
        sample["process_total"] = totals
        self.history[-1] = sample  # replace the entry `sample()` just appended
        return sample

    def sample(self, pid: int | None = None) -> dict[str, Any]:
        disks = {}
        for path in self.disk_paths:
            try:
                usage = shutil.disk_usage(path)
                disks[path] = {"total": usage.total, "free": usage.free}
            except OSError:
                continue

        sample = {
            "ts": time.time(),
            "cpu": self.cpu.sample(),
            "mem": read_memory(),
            "gpus": self.gpu.sample(),
            "process": self.proc.sample(pid),
            "load": read_loadavg(),
            "cpu_temp": read_cpu_temp(),
            "disk": disks,
        }
        self.history.append(sample)
        return sample

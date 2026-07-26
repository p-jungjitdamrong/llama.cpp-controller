"""GPU sampling across vendors.

Each backend reports a list of devices in one common shape so the dashboard does
not care what hardware is underneath:

    {index, vendor, name, busy, mem_used, mem_total, mem_label,
     temp, power, clock_mhz, extra: {...}}

`busy` and any other field may be None when the driver does not expose it.
Backends are probed once at startup; unavailable ones simply never appear.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

DRM_ROOT = Path("/sys/class/drm")

VENDOR_NAMES = {
    "0x1002": "AMD", "0x1022": "AMD", "0x10de": "NVIDIA",
    "0x8086": "Intel", "0x1af4": "Virtio", "0x15ad": "VMware",
}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _first_glob(root: Path, pattern: str) -> Path | None:
    try:
        return next(iter(sorted(root.glob(pattern))), None)
    except OSError:
        return None


def _pci_name(slot: str) -> str | None:
    """Human-readable device name from lspci, when it happens to be installed.

    `lspci -mm` prints shell-quoted fields: slot, class, vendor, device, …
    """
    if not shutil.which("lspci"):
        return None
    try:
        out = subprocess.run(["lspci", "-mm", "-s", slot], capture_output=True,
                             text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        fields = shlex.split(out.strip())
    except ValueError:
        return None
    if len(fields) < 4:
        return None
    vendor, device = fields[2], fields[3]
    vendor_short = vendor.split(",")[0].replace("Advanced Micro Devices", "AMD")
    return f"{vendor_short} {device}".strip()


_FDINFO_SIZE = re.compile(r"^(\d+)\s*([KMG]iB)?$")
_UNITS = {None: 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def read_process_gpu(pid: int) -> dict[str, Any] | None:
    """Per-process GPU usage from DRM fdinfo (amdgpu, i915, xe).

    A process opens the device more than once and every fd reports the same
    totals, so entries are counted once per `drm-client-id`. Engine counters are
    cumulative nanoseconds — the caller turns them into a percentage.
    """
    directory = Path(f"/proc/{pid}/fdinfo")
    try:
        entries = list(directory.iterdir())
    except OSError:
        return None

    seen: set[str] = set()
    vram = gtt = 0
    engines: dict[str, int] = {}
    driver = ""

    for entry in entries:
        try:
            text = entry.read_text()
        except OSError:
            continue
        if "drm-driver" not in text:
            continue
        fields: dict[str, str] = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

        client = fields.get("drm-client-id", entry.name)
        if client in seen:
            continue
        seen.add(client)
        driver = driver or fields.get("drm-driver", "")

        def size(key: str) -> int:
            match = _FDINFO_SIZE.match(fields.get(key, ""))
            if not match:
                return 0
            return int(match.group(1)) * _UNITS.get(match.group(2), 1)

        vram += size("drm-memory-vram") or size("drm-resident-vram")
        gtt += size("drm-memory-gtt") or size("drm-resident-gtt")
        for key, value in fields.items():
            if key.startswith("drm-engine-"):
                try:
                    engines[key[len("drm-engine-"):]] = engines.get(
                        key[len("drm-engine-"):], 0) + int(value.split()[0])
                except (ValueError, IndexError):
                    continue

    if not seen:
        return None
    return {"driver": driver, "vram": vram, "gtt": gtt, "engine_ns": engines}


class GpuBackend:
    name = "generic"

    def sample(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class DrmSysfsBackend(GpuBackend):
    """AMD and Intel GPUs through /sys/class/drm — no tools, no root."""

    name = "drm-sysfs"

    def __init__(self) -> None:
        self.cards: list[dict[str, Any]] = []
        for card in sorted(DRM_ROOT.glob("card[0-9]*")):
            if "-" in card.name:  # card1-DP-1 is a connector, not a device
                continue
            device = card / "device"
            uevent = _read_text(device / "uevent") or ""
            driver = ""
            slot = ""
            for line in uevent.splitlines():
                if line.startswith("DRIVER="):
                    driver = line.split("=", 1)[1]
                elif line.startswith("PCI_SLOT_NAME="):
                    slot = line.split("=", 1)[1]
            if driver not in ("amdgpu", "radeon", "i915", "xe"):
                continue
            vendor_id = _read_text(device / "vendor") or ""
            self.cards.append({
                "path": device,
                "driver": driver,
                "vendor": VENDOR_NAMES.get(vendor_id, vendor_id or "unknown"),
                "name": (_read_text(device / "product_name")
                         or (_pci_name(slot) if slot else None)
                         or f"{VENDOR_NAMES.get(vendor_id, 'GPU')} ({driver})"),
                "hwmon": _first_glob(device, "hwmon/hwmon*"),
            })

    @property
    def available(self) -> bool:
        return bool(self.cards)

    def sample(self) -> list[dict[str, Any]]:
        out = []
        for index, card in enumerate(self.cards):
            device: Path = card["path"]
            entry: dict[str, Any] = {
                "index": index,
                "vendor": card["vendor"],
                "name": card["name"],
                "driver": card["driver"],
                "busy": _read_int(device / "gpu_busy_percent"),
                "mem_used": None,
                "mem_total": None,
                "mem_label": "VRAM",
                "temp": None,
                "power": None,
                "clock_mhz": None,
                "extra": {},
            }

            vram_total = _read_int(device / "mem_info_vram_total")
            if vram_total:
                entry["mem_total"] = vram_total
                entry["mem_used"] = _read_int(device / "mem_info_vram_used")
                gtt_total = _read_int(device / "mem_info_gtt_total")
                gtt_used = _read_int(device / "mem_info_gtt_used")
                if gtt_total:
                    entry["extra"]["gtt_total"] = gtt_total
                    entry["extra"]["gtt_used"] = gtt_used
                    # On an APU the weights live in GTT (shared RAM), and the
                    # 512 MB VRAM carve-out says nothing useful — show GTT.
                    if vram_total <= 2 * 1024**3 and (gtt_used or 0) > (entry["mem_used"] or 0):
                        entry["mem_total"], entry["mem_used"] = gtt_total, gtt_used
                        entry["mem_label"] = "GTT"

            hwmon: Path | None = card["hwmon"]
            if hwmon:
                temp = _read_int(hwmon / "temp1_input")
                power = _read_int(hwmon / "power1_average") or _read_int(hwmon / "power1_input")
                freq = _read_int(hwmon / "freq1_input")
                if temp is not None:
                    entry["temp"] = round(temp / 1000, 1)
                if power is not None:
                    entry["power"] = round(power / 1_000_000, 1)
                if freq is not None:
                    entry["clock_mhz"] = round(freq / 1_000_000)

            if entry["clock_mhz"] is None:  # Intel exposes frequency elsewhere
                for candidate in ("gt_cur_freq_mhz", "gt/gt0/rps_cur_freq_mhz"):
                    value = _read_int(device / candidate)
                    if value is not None:
                        entry["clock_mhz"] = value
                        break
            out.append(entry)
        return out


class NvidiaSmiBackend(GpuBackend):
    """NVIDIA through nvidia-smi; one subprocess per sample, off the event loop."""

    name = "nvidia-smi"
    FIELDS = ("index,name,utilization.gpu,memory.used,memory.total,"
              "temperature.gpu,power.draw,clocks.sm")

    def __init__(self) -> None:
        self.binary = shutil.which("nvidia-smi")
        self.available = bool(self.binary) and bool(self.sample())

    def sample(self) -> list[dict[str, Any]]:
        if not self.binary:
            return []
        try:
            proc = subprocess.run(
                [self.binary, f"--query-gpu={self.FIELDS}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if proc.returncode != 0:
            return []

        def number(raw: str) -> float | None:
            try:
                return float(raw)
            except ValueError:
                return None

        out = []
        for line in proc.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8:
                continue
            used, total = number(cols[3]), number(cols[4])
            out.append({
                "index": int(cols[0]) if cols[0].isdigit() else len(out),
                "vendor": "NVIDIA",
                "name": cols[1],
                "driver": "nvidia",
                "busy": number(cols[2]),
                "mem_used": int(used * 1024**2) if used is not None else None,
                "mem_total": int(total * 1024**2) if total is not None else None,
                "mem_label": "VRAM",
                "temp": number(cols[5]),
                "power": number(cols[6]),
                "clock_mhz": number(cols[7]),
                "extra": {},
            })
        return out


def detect_backends() -> list[GpuBackend]:
    backends: list[GpuBackend] = []
    drm = DrmSysfsBackend()
    if drm.available:
        backends.append(drm)
    nvidia = NvidiaSmiBackend()
    if nvidia.available:
        backends.append(nvidia)
    return backends


class GpuMonitor:
    """All detected GPUs, re-indexed so the UI has stable ids."""

    def __init__(self) -> None:
        self.backends = detect_backends()

    @property
    def available(self) -> bool:
        return bool(self.backends)

    def describe(self) -> list[dict[str, Any]]:
        return [{"index": g["index"], "vendor": g["vendor"], "name": g["name"],
                 "driver": g.get("driver", "")} for g in self.sample()]

    def process_memory(self) -> dict[int, int]:
        """pid -> VRAM bytes, for drivers that only report it centrally."""
        out: dict[int, int] = {}
        for backend in self.backends:
            if not isinstance(backend, NvidiaSmiBackend) or not backend.binary:
                continue
            try:
                proc = subprocess.run(
                    [backend.binary, "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        out[int(parts[0])] = int(float(parts[1]) * 1024**2)
                    except ValueError:
                        continue
        return out

    def sample(self) -> list[dict[str, Any]]:
        devices: list[dict[str, Any]] = []
        for backend in self.backends:
            try:
                for device in backend.sample():
                    device["index"] = len(devices)
                    device["backend"] = backend.name
                    devices.append(device)
            except Exception:  # a flaky backend must not stop the others
                continue
        return devices

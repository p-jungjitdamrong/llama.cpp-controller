"""Discovery of local GGUF models with cached header metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .gguf import read_metadata

MAX_DEPTH = 3
_QUANT_RE = re.compile(
    r"(IQ\d[_A-Z]*|Q\d(_[01KMSL])*|BF16|F16|F32|TQ\d_\d)", re.IGNORECASE
)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])")

# Split multi-part downloads (…-00001-of-00003.gguf) so only the first shows up.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _walk(root: Path, depth: int = 0) -> Iterable[Path]:
    if depth > MAX_DEPTH or not root.is_dir():
        return
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink() and entry.resolve().is_dir():
                continue
            if entry.is_dir():
                if entry.name.startswith(".") and entry.name != ".cache":
                    continue
                yield from _walk(entry, depth + 1)
            elif entry.suffix.lower() == ".gguf":
                yield entry
        except OSError:
            continue


def _guess_from_name(name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    quant = _QUANT_RE.search(name)
    if quant:
        out["quant_guess"] = quant.group(0).upper()
    size = _SIZE_RE.search(name)
    if size:
        out["size_guess"] = f"{size.group(1)}B"
    return out


class ModelIndex:
    """Scans the configured directories; GGUF headers are parsed once per file."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    def scan(self, dirs: Iterable[Path]) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for directory in dirs:
            for path in _walk(directory):
                resolved = str(path.resolve())
                if resolved in found:
                    continue
                shard = _SHARD_RE.search(path.name)
                if shard and shard.group(1) != "00001":
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                key = (resolved, stat.st_size, int(stat.st_mtime))
                entry = self._cache.get(key)
                if entry is None:
                    entry = {
                        **_guess_from_name(path.name),
                        **read_metadata(path),
                        "path": resolved,
                        "name": path.name,
                        "dir": str(path.parent),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "sharded": bool(shard),
                    }
                    self._cache[key] = entry
                # vocab-only files (llama.cpp test fixtures) carry no tensors
                if entry.get("tensor_count") == 0:
                    continue
                found[resolved] = entry
        return sorted(found.values(), key=lambda m: m["name"].lower())

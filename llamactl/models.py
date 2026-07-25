"""Discovery of local GGUF models with cached header metadata."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from .gguf import read_metadata

MAX_DEPTH = 5
# directories that never hold models but are expensive to walk
PRUNE = {
    "node_modules", "site-packages", "dist-packages", ".git", ".venv", "venv",
    "__pycache__", "blobs", "Trash", "snap",
}

_QUANT_RE = re.compile(
    r"(IQ\d[_A-Z]*|Q\d(_[01KMSL])*|BF16|F16|F32|TQ\d_\d|PQ\d_\d)", re.IGNORECASE
)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])")
# split multi-part downloads (…-00001-of-00003.gguf) so only the first shows up
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
# huggingface cache layout: hub/models--<org>--<name>/snapshots/<rev>/file.gguf
_HF_REPO_RE = re.compile(r"models--([^/]+)")


def _walk(root: Path, depth: int = 0) -> Iterable[Path]:
    if depth > MAX_DEPTH or not root.is_dir():
        return
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir():
                if entry.is_symlink():
                    continue  # symlinked directories invite loops
                if entry.name in PRUNE:
                    continue
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


def _hf_repo(path: Path) -> str | None:
    match = _HF_REPO_RE.search(str(path))
    if not match:
        return None
    return match.group(1).replace("--", "/")


class ModelIndex:
    """Scans the configured directories; GGUF headers are parsed once per file."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    def scan(self, dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Return (usable models, skipped files with the reason they were skipped)."""
        found: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()

        for directory in dirs:
            for path in _walk(directory):
                try:
                    # follows symlinks — the huggingface cache is a tree of them
                    stat = path.stat()
                except OSError:
                    skipped.append({"name": path.name, "reason": "unreadable"})
                    continue
                # dedupe on the real file, but keep the readable path for launching
                real = str(path.resolve())
                if real in seen:
                    continue
                seen.add(real)

                shard = _SHARD_RE.search(path.name)
                if shard and shard.group(1) != "00001":
                    continue
                if not os.access(path, os.R_OK):
                    skipped.append({"name": path.name, "reason": "no read permission"})
                    continue

                key = (real, stat.st_size, int(stat.st_mtime))
                entry = self._cache.get(key)
                if entry is None:
                    entry = {
                        **_guess_from_name(path.name),
                        **read_metadata(path),
                        "path": str(path),
                        "name": path.name,
                        "dir": str(path.parent),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "sharded": bool(shard),
                        "repo": _hf_repo(path),
                    }
                    self._cache[key] = entry

                if entry.get("error"):
                    skipped.append({"name": path.name, "reason": entry["error"]})
                    continue
                # vocab-only fixtures shipped with llama.cpp carry no tensors
                if entry.get("tensor_count") == 0:
                    skipped.append({"name": path.name, "reason": "vocab-only, no tensors"})
                    continue
                # multimodal projectors are passed via --mmproj, not -m
                if "mmproj" in path.name.lower() or entry.get("architecture") == "clip":
                    skipped.append({"name": path.name, "reason": "mmproj projector"})
                    continue

                found[str(path)] = entry

        return sorted(found.values(), key=lambda m: m["name"].lower()), skipped

"""Browsing and downloading GGUF models from the Hugging Face Hub.

Only the public read API is used — no token, no `huggingface_hub` dependency.
Downloads stream to `<name>.gguf.part` and are renamed on completion, so an
interrupted transfer can resume with a Range request instead of starting over.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import httpx

API = "https://huggingface.co/api"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
CHUNK = 1 << 20  # 1 MiB

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


async def search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Repos matching `query` that ship GGUF files, most downloaded first."""
    params = {
        "search": query,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(f"{API}/models", params=params)
        response.raise_for_status()
        data = response.json()
    return [
        {
            "id": item["id"],
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "updated": item.get("createdAt", ""),
            "tags": [t for t in item.get("tags", []) if t in ("gguf", "conversational", "text-generation")],
        }
        for item in data
    ]


async def list_files(repo: str) -> list[dict[str, Any]]:
    """GGUF files in a repo, including ones inside quant subdirectories."""
    url = f"{API}/models/{repo}/tree/main"
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url, params={"recursive": "true"})
        response.raise_for_status()
        entries = response.json()
    files = [
        {
            "path": entry["path"],
            "name": entry["path"].rsplit("/", 1)[-1],
            "size": entry.get("size") or (entry.get("lfs") or {}).get("size") or 0,
            "is_projector": "mmproj" in entry["path"].lower(),
        }
        for entry in entries
        if entry.get("type") == "file" and entry["path"].lower().endswith(".gguf")
    ]
    return sorted(files, key=lambda f: f["size"])


class Download:
    """One file transfer, tracked so the dashboard can show progress."""

    _counter = 0

    def __init__(self, repo: str, path: str, dest: Path) -> None:
        Download._counter += 1
        self.id = Download._counter
        self.repo = repo
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self.dest = dest
        self.part = dest.with_suffix(dest.suffix + ".part")
        self.total = 0
        self.downloaded = 0
        self.status = "queued"
        self.error = ""
        self.started_at = time.time()
        self.speed = 0.0
        self.task: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        percent = round(100.0 * self.downloaded / self.total, 1) if self.total else 0.0
        remaining = None
        if self.status == "downloading" and self.speed > 0 and self.total:
            remaining = round((self.total - self.downloaded) / self.speed)
        return {
            "id": self.id,
            "repo": self.repo,
            "name": self.name,
            "dest": str(self.dest),
            "total": self.total,
            "downloaded": self.downloaded,
            "percent": percent,
            "status": self.status,
            "error": self.error,
            "speed": round(self.speed),
            "eta": remaining,
        }


class Downloader:
    def __init__(self, dest_dir: Path) -> None:
        self.dest_dir = dest_dir
        self.jobs: dict[int, Download] = {}

    def public(self) -> list[dict[str, Any]]:
        return [job.public() for job in sorted(self.jobs.values(), key=lambda j: -j.id)]

    def start(self, repo: str, path: str) -> Download:
        safe = _SAFE_NAME.sub("_", path.rsplit("/", 1)[-1])
        if not safe.lower().endswith(".gguf"):
            raise ValueError("only .gguf files can be downloaded")
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self.dest_dir / safe
        if dest.exists():
            raise FileExistsError(f"{dest} already exists")
        for job in self.jobs.values():
            if job.dest == dest and job.status in ("queued", "downloading"):
                raise FileExistsError(f"{safe} is already downloading")

        job = Download(repo, path, dest)
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job))
        return job

    def cancel(self, job_id: int) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.task is None or job.status not in ("queued", "downloading"):
            return False
        job.task.cancel()
        return True

    def forget(self, job_id: int) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in ("queued", "downloading"):
            return False
        del self.jobs[job_id]
        return True

    async def _run(self, job: Download) -> None:
        url = RESOLVE.format(repo=job.repo, path=job.path)
        resume_from = job.part.stat().st_size if job.part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        job.status = "downloading"
        job.downloaded = resume_from
        last_time, last_bytes = time.monotonic(), resume_from

        try:
            timeout = httpx.Timeout(30.0, read=120.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416:  # already complete on disk
                        job.part.rename(job.dest)
                        job.status = "done"
                        return
                    response.raise_for_status()
                    if response.status_code == 200:  # server ignored the range
                        resume_from = job.downloaded = 0
                    length = int(response.headers.get("content-length", 0))
                    job.total = length + resume_from

                    mode = "ab" if resume_from else "wb"
                    with job.part.open(mode) as handle:
                        async for chunk in response.aiter_bytes(CHUNK):
                            handle.write(chunk)
                            job.downloaded += len(chunk)
                            now = time.monotonic()
                            if now - last_time >= 0.5:
                                rate = (job.downloaded - last_bytes) / (now - last_time)
                                # smooth it so the UI does not jitter
                                job.speed = rate if not job.speed else job.speed * 0.7 + rate * 0.3
                                last_time, last_bytes = now, job.downloaded

            job.part.rename(job.dest)
            job.status = "done"
            job.speed = 0.0
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except (httpx.HTTPError, OSError) as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"

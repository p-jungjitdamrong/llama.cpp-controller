"""Try a few launch configurations for one model and keep the best.

The point is not a leaderboard — it is a saved preset. Each candidate is started
for real through the supervisor, asked the same question, and timed; the winner
can be written back as that model's settings, which then apply both when it is
started on its own and when a router loads it.

Two knobs are swept by default because they are the two that actually moved the
numbers on modest hardware: GPU offload, which is not automatically faster on an
integrated GPU, and thread count, which matters most when the CPU is doing the
work.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Config, LaunchParams

# long enough that prompt processing is measurable, short enough to stay cheap
PROMPT = (
    "You are given a short brief. Read it and answer in plain prose.\n\n"
    "Brief: a small team runs language models on a single laptop with an "
    "integrated GPU and limited memory. They want to understand the trade-offs "
    "between running more models at once and giving each model more context, "
    "and they care about the time between pressing enter and seeing the first "
    "word of the answer.\n\n"
    "Question: in two or three sentences, what should they measure first?"
)
MAX_TOKENS = 120


def physical_cores() -> int:
    """Cores rather than hyperthreads — llama.cpp usually prefers the former."""
    try:
        ids = set()
        core, package = None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif line.startswith("physical id"):
                package = line.split(":")[1].strip()
            elif not line.strip() and core is not None:
                ids.add((package, core))
                core, package = None, None
        if core is not None:
            ids.add((package, core))
        if ids:
            return len(ids)
    except OSError:
        pass
    import os
    return max(1, (os.cpu_count() or 2) // 2)


def default_candidates(base: LaunchParams) -> list[dict[str, Any]]:
    """A 2x2 over the knobs worth sweeping, described for the UI."""
    cores = physical_cores()
    return [
        {"label": "GPU, auto threads", "n_gpu_layers": 99, "threads": 0},
        {"label": "CPU, auto threads", "n_gpu_layers": 0, "threads": 0},
        {"label": f"GPU, {cores} threads", "n_gpu_layers": 99, "threads": cores},
        {"label": f"CPU, {cores} threads", "n_gpu_layers": 0, "threads": cores},
    ]


class BenchRun:
    def __init__(self, run_id: str, model_path: str, total: int) -> None:
        self.id = run_id
        self.model_path = model_path
        self.total = total
        self.cancelled = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.started_at = time.time()
        self.results: list[dict[str, Any]] = []


class Tuner:
    """Runs one benchmark at a time and remembers the results per model."""

    def __init__(self, cfg: Config, pool: Any, publish: Callable[[dict], None]) -> None:
        self.cfg = cfg
        self.pool = pool
        self.publish = publish
        self.run: BenchRun | None = None

    @property
    def busy(self) -> bool:
        return self.run is not None

    def _emit(self, event: str, **data: Any) -> None:
        run = self.run
        self.publish({"event": event, "run_id": run.id if run else None,
                      "model_path": run.model_path if run else None,
                      "ts": time.time(), **data})

    def start(self, model_path: str, port: int,
              candidates: list[dict[str, Any]] | None = None,
              ctx_size: int | None = None,
              stop_running: bool = False) -> BenchRun:
        if self.busy:
            raise RuntimeError("a benchmark is already running")
        model = Path(model_path).expanduser()
        if not model.is_file():
            raise FileNotFoundError(f"model not found: {model}")

        # each candidate starts the same model, which the pool refuses while it
        # is already serving — so say so once, up front, instead of failing on
        # every configuration in turn
        serving = self.pool.by_model(str(model))
        restore = None
        if serving is not None and serving.pid:
            if not stop_running:
                raise RuntimeError(
                    f"{model.name} is serving on port {serving.params.port}. "
                    f"Tuning restarts it several times — stop it first, or let "
                    f"the benchmark stop it and put it back afterwards."
                )
            restore = (serving.id, serving.params)

        base = self.cfg.params_for(str(model))
        if ctx_size:
            base.ctx_size = int(ctx_size)
        options = candidates or default_candidates(base)
        run = BenchRun(uuid.uuid4().hex[:12], str(model), len(options))
        self.run = run
        run.task = asyncio.create_task(self._sweep(run, base, options, port, restore))
        return run

    def cancel(self, run_id: str) -> bool:
        if self.run and self.run.id == run_id:
            self.run.cancelled.set()
            return True
        return False

    async def _sweep(self, run: BenchRun, base: LaunchParams,
                     options: list[dict[str, Any]], port: int,
                     restore: tuple[int, LaunchParams] | None = None) -> None:
        self._emit("start", model=Path(run.model_path).name, total=run.total,
                   ctx_size=base.ctx_size, will_restore=bool(restore))
        try:
            if restore is not None:
                self._emit("note", message="stopping the running copy for the sweep")
                with contextlib.suppress(Exception):
                    await self.pool.stop(restore[0])
                    await self.pool.remove(restore[0])
            for index, option in enumerate(options, 1):
                if run.cancelled.is_set():
                    self._emit("cancelled", done=index - 1)
                    return
                label = option.get("label") or f"config {index}"
                params = LaunchParams.from_dict({**asdict(base), **option, "port": port,
                                                 "host": "127.0.0.1"})
                self._emit("candidate", index=index, label=label,
                           params={"n_gpu_layers": params.n_gpu_layers,
                                   "threads": params.threads,
                                   "ctx_size": params.ctx_size})
                result = await self._measure(run, params, label)
                run.results.append(result)
                self._emit("result", index=index, **result)

            chat = [r for r in run.results if r.get("generate_tps")]
            embed = [r for r in run.results if r.get("embed_ms")]
            if chat:
                best = max(chat, key=lambda r: r["generate_tps"])
            elif embed:
                best = min(embed, key=lambda r: r["embed_ms"])   # latency: less is better
            else:
                best = None
            record = {"ran_at": time.time(), "ctx_size": base.ctx_size,
                      "results": run.results, "best": best}
            self.cfg.benchmarks[run.model_path] = record
            self.cfg.save()
            self._emit("done", best=best, results=run.results,
                       seconds=round(time.time() - run.started_at, 1))
        except Exception as exc:
            self._emit("error", message=f"{type(exc).__name__}: {exc}")
        finally:
            if restore is not None:
                # put back what was serving before the sweep borrowed the model
                try:
                    status = await self.pool.start(run.model_path, restore[1])
                    self._emit("note", message=f"restarted on port {status['bind_port']}")
                except Exception as exc:
                    self._emit("note", message=f"could not restart it: {exc}")
            self.run = None

    @staticmethod
    def is_embedding_model(model_path: str, params: LaunchParams) -> bool:
        """Embedding servers have no chat endpoint — they need a different probe."""
        if "--embedding" in (params.extra_args or ""):
            return True
        from .gguf import read_metadata
        arch = (read_metadata(Path(model_path)).get("architecture") or "").lower()
        return "embedding" in arch or "bert" in arch

    async def _probe_embedding(self, url: str, client: httpx.AsyncClient) -> dict[str, Any]:
        """Median latency over a few identical requests, plus tokens/s if reported."""
        timings = []
        tokens = None
        for _ in range(3):
            started = time.monotonic()
            response = await client.post(f"{url}/v1/embeddings", json={"input": PROMPT})
            response.raise_for_status()
            timings.append((time.monotonic() - started) * 1000)
            usage = response.json().get("usage") or {}
            tokens = usage.get("prompt_tokens") or tokens
        median = sorted(timings)[len(timings) // 2]
        row: dict[str, Any] = {"mode": "embedding", "embed_ms": round(median, 1)}
        if tokens:
            row["prompt_tps"] = round(tokens / (median / 1000), 1)
        return row

    async def _measure(self, run: BenchRun, params: LaunchParams,
                       label: str) -> dict[str, Any]:
        """Start the model with these params, time one answer, stop it again."""
        row: dict[str, Any] = {"label": label, "n_gpu_layers": params.n_gpu_layers,
                               "threads": params.threads}
        instance_id = None
        started = time.monotonic()
        try:
            status = await self.pool.start(run.model_path, params)
            instance_id = status["id"]
            instance = self.pool.get(instance_id)
            deadline = time.monotonic() + 300
            while instance.state.value == "starting" and time.monotonic() < deadline:
                if run.cancelled.is_set():
                    return {**row, "error": "cancelled"}
                await asyncio.sleep(1)
            if instance.state.value != "ready":
                return {**row, "error": instance.last_error or "failed to load"}
            row["load_seconds"] = round(time.monotonic() - started, 1)

            async with httpx.AsyncClient(timeout=httpx.Timeout(20, read=300)) as client:
                if self.is_embedding_model(run.model_path, params):
                    row.update(await self._probe_embedding(instance.base_url, client))
                else:
                    response = await client.post(
                        f"{instance.base_url}/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": PROMPT}],
                              "max_tokens": MAX_TOKENS, "temperature": 0.0})
                    response.raise_for_status()
                    timings = response.json().get("timings") or {}
                    row["mode"] = "chat"
                    row["prompt_tps"] = round(timings.get("prompt_per_second") or 0, 1)
                    row["generate_tps"] = round(timings.get("predicted_per_second") or 0, 1)
                    row["tokens"] = timings.get("predicted_n")
        except MemoryError as exc:
            row["error"] = str(exc)
        except httpx.HTTPStatusError as exc:
            row["error"] = (f"the server answered {exc.response.status_code} — this model "
                            f"may not support that endpoint")
        except (httpx.HTTPError, RuntimeError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:160]
        finally:
            if instance_id is not None:
                with_suppress = getattr(self.pool, "stop", None)
                if with_suppress:
                    try:
                        await self.pool.stop(instance_id)
                        await self.pool.remove(instance_id)
                    except Exception:
                        pass
        return row

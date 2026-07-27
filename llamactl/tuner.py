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


MAX_CANDIDATES = 32
SWEEPABLE = ("n_gpu_layers", "threads", "ctx_size", "batch_size", "parallel", "flash_attn")


def default_candidates(base: LaunchParams) -> list[dict[str, Any]]:
    """A 2x2 over the knobs worth sweeping, described for the UI."""
    cores = physical_cores()
    return [
        {"label": "GPU, auto threads", "n_gpu_layers": 99, "threads": 0},
        {"label": "CPU, auto threads", "n_gpu_layers": 0, "threads": 0},
        {"label": f"GPU, {cores} threads", "n_gpu_layers": 99, "threads": cores},
        {"label": f"CPU, {cores} threads", "n_gpu_layers": 0, "threads": cores},
    ]


def offload_steps(model_path: str) -> list[int]:
    """Offload values worth trying for this model, including partial ones.

    -ngl is a number of layers, not a switch: on a small shared GPU the best
    setting is often part of the model on the GPU and the rest on the CPU.
    """
    from .gguf import read_metadata

    layers = read_metadata(Path(model_path)).get("n_layer") or 32
    quarters = [0, layers // 4, layers // 2, (layers * 3) // 4, layers]
    return sorted({value for value in quarters if 0 <= value <= layers})


def suggested_sweep(model_path: str, base: LaunchParams) -> dict[str, list[Any]]:
    return {"n_gpu_layers": offload_steps(model_path),
            "threads": sorted({0, physical_cores()})}


def _describe(key: str, value: Any) -> str:
    if key == "n_gpu_layers":
        return "CPU" if value == 0 else f"{value} GPU"
    if key == "threads":
        return "auto threads" if value in (0, None) else f"{value}t"
    if key == "ctx_size":
        return f"{value // 1024}k ctx" if value % 1024 == 0 else f"{value} ctx"
    if key == "batch_size":
        return "default batch" if value in (0, None) else f"batch {value}"
    if key == "parallel":
        return f"{value} slot" + ("" if value == 1 else "s")
    if key == "flash_attn":
        return f"fa {value}"
    return f"{key} {value}"


def build_candidates(sweep: dict[str, list[Any]], base: LaunchParams) -> list[dict[str, Any]]:
    """Cross product of the values chosen for each knob.

    Only knobs with more than one value appear in the label, so the table reads
    as the thing that varies rather than a wall of identical text.
    """
    import itertools

    axes = {key: list(dict.fromkeys(values))          # de-duplicate, keep order
            for key, values in sweep.items()
            if key in SWEEPABLE and isinstance(values, list) and values}
    if not axes:
        return default_candidates(base)

    varying = [key for key, values in axes.items() if len(values) > 1] or list(axes)
    combos = []
    for values in itertools.product(*axes.values()):
        option = dict(zip(axes.keys(), values))
        option["label"] = ", ".join(_describe(key, option[key]) for key in varying)
        combos.append(option)
        if len(combos) > MAX_CANDIDATES:
            raise ValueError(
                f"that is more than {MAX_CANDIDATES} configurations — "
                f"drop a value or sweep one knob at a time")
    return combos


def _median(values: list[float]) -> float:
    ordered = sorted(v for v in values if v)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


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
              stop_running: bool = False,
              repeats: int = 1) -> BenchRun:
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
        run.task = asyncio.create_task(
            self._sweep(run, base, options, port, restore, max(1, min(9, int(repeats)))))
        return run

    def cancel(self, run_id: str) -> bool:
        if self.run and self.run.id == run_id:
            self.run.cancelled.set()
            return True
        return False

    async def _sweep(self, run: BenchRun, base: LaunchParams,
                     options: list[dict[str, Any]], port: int,
                     restore: tuple[int, LaunchParams] | None = None,
                     repeats: int = 1) -> None:
        self._emit("start", model=Path(run.model_path).name, total=run.total,
                   ctx_size=base.ctx_size, will_restore=bool(restore),
                   repeats=repeats)
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
                result = await self._measure(run, params, label, repeats)
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

    async def _probe_embedding(self, url: str, client: httpx.AsyncClient,
                               repeats: int = 3) -> dict[str, Any]:
        """Median latency over a few identical requests, plus tokens/s if reported."""
        timings = []
        tokens = None
        for _ in range(max(3, repeats)):
            started = time.monotonic()
            response = await client.post(f"{url}/v1/embeddings", json={"input": PROMPT})
            response.raise_for_status()
            timings.append((time.monotonic() - started) * 1000)
            usage = response.json().get("usage") or {}
            tokens = usage.get("prompt_tokens") or tokens
        middle = _median(timings)
        row: dict[str, Any] = {"mode": "embedding", "runs": len(timings),
                               "embed_ms": round(middle, 1),
                               "spread": [round(min(timings), 1), round(max(timings), 1)]}
        if tokens and middle:
            row["prompt_tps"] = round(tokens / (middle / 1000), 1)
        return row

    async def _measure(self, run: BenchRun, params: LaunchParams,
                       label: str, repeats: int = 1) -> dict[str, Any]:
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
                    row.update(await self._probe_embedding(instance.base_url, client, repeats))
                else:
                    prompts, generates = [], []
                    for attempt in range(repeats):
                        if run.cancelled.is_set():
                            break
                        # a different opening each time, or llama.cpp would reuse
                        # the cached prefix and report an unreal prompt speed
                        text = f"Round {attempt + 1}. {PROMPT}"
                        response = await client.post(
                            f"{instance.base_url}/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": text}],
                                  "max_tokens": MAX_TOKENS, "temperature": 0.0})
                        response.raise_for_status()
                        timings = response.json().get("timings") or {}
                        prompts.append(timings.get("prompt_per_second") or 0)
                        generates.append(timings.get("predicted_per_second") or 0)
                        row["tokens"] = timings.get("predicted_n")
                    row["mode"] = "chat"
                    row["runs"] = len(generates)
                    row["prompt_tps"] = round(_median(prompts), 1)
                    row["generate_tps"] = round(_median(generates), 1)
                    if len(generates) > 1:
                        row["spread"] = [round(min(generates), 1), round(max(generates), 1)]
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

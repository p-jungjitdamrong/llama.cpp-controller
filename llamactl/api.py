"""HTTP + websocket API and static hosting for the dashboard."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __author__, __url__, __version__, hub
from .config import Config, LaunchParams
from .hub import Downloader
from .metrics import Monitor
from .models import ModelIndex
from .supervisor import ServerInstance, SupervisorPool, find_external_servers
from .tuner import Tuner, build_candidates, default_candidates, suggested_sweep

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(cfg: Config) -> FastAPI:
    monitor = Monitor(history_size=cfg.history_size)
    index = ModelIndex()
    pool = SupervisorPool(cfg)
    downloader = Downloader(cfg.download_dir())
    clients: set[WebSocket] = set()

    def broadcast(kind: str, data: Any) -> None:
        payload = json.dumps({"type": kind, "data": data})
        for ws in list(clients):
            asyncio.create_task(_push(ws, payload))

    async def _push(ws: WebSocket, payload: str) -> None:
        try:
            await ws.send_text(payload)
        except Exception:
            clients.discard(ws)

    tuner = Tuner(cfg, pool, lambda event: broadcast("bench", event))

    def instance_or_404(instance_id: Any) -> ServerInstance:
        try:
            instance = pool.get(int(instance_id))
        except (TypeError, ValueError):
            instance = None
        if instance is None:
            raise HTTPException(404, f"no instance with id {instance_id}")
        return instance

    async def sampler() -> None:
        """One 1 Hz sampling loop shared by every connected client."""
        while True:
            try:
                # off the event loop: some GPU backends shell out to a tool
                sample = await asyncio.to_thread(monitor.sample_many, pool.pids)
                servers = pool.status()
                for status in servers:
                    status["process"] = sample["processes"].get(status["pid"]) if status["pid"] else None
                sample["servers"] = servers
                sample["downloads"] = downloader.public()
                payload = json.dumps({"type": "metrics", "data": sample})
                for ws in list(clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        clients.discard(ws)
            except Exception as exc:  # keep the loop alive whatever happens
                print(f"[sampler] {exc!r}")
            await asyncio.sleep(1.0)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(sampler())
        # in the background, so the dashboard is reachable while models load
        boot = asyncio.create_task(pool.autostart(cfg.autostart))
        try:
            yield
        finally:
            for pending in (task, boot):
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
            await pool.shutdown()

    app = FastAPI(title="llama-controller", docs_url="/api/docs", lifespan=lifespan)

    # ------------------------------------------------------------------ auth

    # The page itself stays open so the browser can load it and ask for a token;
    # everything that reads state or changes it is behind the check.
    PUBLIC_PREFIXES = ("/static", "/favicon")

    def token_of(request: Request) -> str:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return request.headers.get("x-api-key", "") or request.query_params.get("token", "")

    def allowed(token: str, host: str) -> bool:
        if not cfg.auth_enabled or not cfg.auth_token:
            return True
        if cfg.auth_allow_localhost and host in ("127.0.0.1", "::1", "localhost"):
            return True
        return secrets.compare_digest(token or "", cfg.auth_token)

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        path = request.url.path
        host = request.client.host if request.client else ""
        if path == "/" or path.startswith(PUBLIC_PREFIXES) or allowed(token_of(request), host):
            return await call_next(request)
        return JSONResponse({"detail": "authentication required"}, status_code=401)

    @app.get("/api/auth")
    async def auth_status() -> dict[str, Any]:
        return {"enabled": cfg.auth_enabled, "has_token": bool(cfg.auth_token),
                "allow_localhost": cfg.auth_allow_localhost}

    @app.post("/api/auth")
    async def auth_configure(payload: dict = Body(...)) -> dict[str, Any]:
        """Turn access control on or off. The token is returned only when minted."""
        token = None
        if "allow_localhost" in payload:
            cfg.auth_allow_localhost = bool(payload["allow_localhost"])
        if payload.get("enabled") is not None:
            cfg.auth_enabled = bool(payload["enabled"])
        if cfg.auth_enabled and (not cfg.auth_token or payload.get("rotate")):
            token = secrets.token_urlsafe(32)
            cfg.auth_token = token
        cfg.save()
        return {"enabled": cfg.auth_enabled, "allow_localhost": cfg.auth_allow_localhost,
                "has_token": bool(cfg.auth_token), "token": token}

    # ------------------------------------------------------------------ meta

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "system": monitor.static_info(),
            "config": cfg.public_dict(),
            "server_bin_exists": cfg.server_bin.is_file(),
            "external_servers": find_external_servers(exclude_pids=pool.pids),
            "download_dir": str(downloader.dest_dir),
            "app": {"version": __version__, "author": __author__, "url": __url__},
        }

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return cfg.public_dict()

    @app.post("/api/config")
    async def set_config(patch: dict = Body(...)) -> dict[str, Any]:
        if "llama_server_bin" in patch:
            binary = Path(patch["llama_server_bin"]).expanduser()
            if not binary.is_file():
                raise HTTPException(400, f"no llama-server at {binary}")
            cfg.llama_server_bin = patch["llama_server_bin"]
        if "model_dirs" in patch:
            dirs = [d for d in patch["model_dirs"] if str(d).strip()]
            if not dirs:
                raise HTTPException(400, "at least one model directory is required")
            missing = [d for d in dirs if not Path(d).expanduser().is_dir()]
            if missing:
                raise HTTPException(400, f"not a directory: {', '.join(missing)}")
            cfg.model_dirs = dirs
            downloader.dest_dir = cfg.download_dir()
        for key in ("history_size", "log_buffer_lines"):
            if key in patch:
                try:
                    setattr(cfg, key, max(10, int(patch[key])))
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{key} must be a number") from None
        if "defaults" in patch:
            cfg.defaults = LaunchParams.from_dict(patch["defaults"])
        cfg.save()
        return cfg.public_dict()

    # ---------------------------------------------------------------- models

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        found, skipped = await asyncio.to_thread(index.scan, cfg.resolved_model_dirs())
        running = {i.model_path: i.params.port for i in pool.instances.values() if i.pid}
        autostarted = {e.get("model_path") for e in cfg.autostart}
        for entry in found:
            entry["params"] = asdict(cfg.params_for(entry["path"]))
            entry["running_port"] = running.get(entry["path"])
            entry["autostart"] = entry["path"] in autostarted
            entry["benchmark"] = cfg.benchmarks.get(entry["path"])
            entry["has_preset"] = entry["path"] in cfg.presets

        dirs = []
        for directory in cfg.resolved_model_dirs():
            info: dict[str, Any] = {"path": str(directory), "exists": directory.is_dir()}
            if info["exists"]:
                with contextlib.suppress(OSError):
                    usage = shutil.disk_usage(directory)
                    info["free"] = usage.free
                    info["total"] = usage.total
                info["size"] = sum(m["size"] for m in found
                                   if m["dir"].startswith(str(directory)))
            dirs.append(info)

        return {
            "models": found,
            "skipped": skipped,
            "dirs": [str(d) for d in cfg.resolved_model_dirs()],
            "dir_info": dirs,
            "download_dir": str(downloader.dest_dir),
        }

    @app.post("/api/models/delete")
    async def delete_model(payload: dict = Body(...)) -> dict[str, Any]:
        """Delete a GGUF file. Only inside a configured directory, never in use."""
        raw = payload.get("path") or ""
        if not payload.get("confirm"):
            raise HTTPException(400, "confirm must be true")
        target = Path(raw).expanduser().resolve()
        if target.suffix.lower() != ".gguf" or not target.is_file():
            raise HTTPException(400, f"not a gguf file: {target}")
        roots = [d.resolve() for d in cfg.resolved_model_dirs()]
        if not any(target == root or root in target.parents for root in roots):
            raise HTTPException(403, "file is outside the configured model directories")
        for instance in pool.instances.values():
            if instance.pid and Path(instance.model_path).resolve() == target:
                raise HTTPException(409,
                                    f"{target.name} is being served on port "
                                    f"{instance.params.port} — stop it first")
        size = target.stat().st_size
        target.unlink()
        pool.log(f"deleted model {target} ({size / 1024**3:.2f} GB)")
        cfg.presets.pop(str(target), None)
        cfg.autostart = [e for e in cfg.autostart
                         if Path(e.get("model_path", "")).resolve() != target]
        cfg.save()
        return {"deleted": str(target), "freed": size}

    # ------------------------------------------------------------ benchmark

    @app.get("/api/bench")
    async def bench_status() -> dict[str, Any]:
        run = tuner.run
        return {
            "running": bool(run),
            "run_id": run.id if run else None,
            "model_path": run.model_path if run else None,
            "results": {path: record for path, record in cfg.benchmarks.items()},
        }

    @app.post("/api/bench/run")
    async def bench_run(payload: dict = Body(...)) -> dict[str, Any]:
        model_path = payload.get("model_path")
        if not model_path:
            raise HTTPException(400, "model_path is required")
        if tuner.busy:
            raise HTTPException(409, "a benchmark is already running")
        # a port of its own so a sweep never disturbs what is already serving
        port = int(payload.get("port") or 8199)
        candidates = payload.get("candidates")
        if not candidates and payload.get("sweep"):
            try:
                candidates = build_candidates(payload["sweep"],
                                              cfg.params_for(model_path), model_path)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        try:
            run = tuner.start(model_path, port, candidates,
                              payload.get("ctx_size"),
                              bool(payload.get("stop_running")),
                              int(payload.get("repeats") or 1))
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            # 409 so the UI can offer to stop it and put it back
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run.id, "total": run.total}

    @app.post("/api/bench/cancel")
    async def bench_cancel(payload: dict = Body(...)) -> dict[str, Any]:
        if not tuner.cancel(payload.get("run_id") or ""):
            raise HTTPException(404, "no such run")
        return {"ok": True}

    @app.post("/api/bench/apply")
    async def bench_apply(payload: dict = Body(...)) -> dict[str, Any]:
        """Save a measured configuration as this model's settings."""
        model_path = payload.get("model_path") or ""
        record = cfg.benchmarks.get(model_path)
        if not record or not record.get("best"):
            raise HTTPException(404, "no benchmark result for that model")
        chosen = payload.get("result") or record["best"]
        params = asdict(cfg.params_for(model_path))
        for key in ("n_gpu_layers", "threads"):
            if chosen.get(key) is not None:
                params[key] = chosen[key]
        if record.get("ctx_size"):
            params["ctx_size"] = record["ctx_size"]
        cfg.presets[model_path] = params
        cfg.save()
        return {"model_path": model_path, "params": params}

    @app.get("/api/bench/candidates")
    async def bench_candidates(model_path: str) -> dict[str, Any]:
        """What to offer in the sweep form, sized to this particular model."""
        base = cfg.params_for(model_path)
        from .gguf import read_metadata
        return {"candidates": default_candidates(base),
                "layers": read_metadata(Path(model_path)).get("n_layer"),
                "suggested": suggested_sweep(model_path, base),
                "current": {"n_gpu_layers": base.n_gpu_layers, "threads": base.threads,
                            "ctx_size": base.ctx_size, "batch_size": base.batch_size},
                "max_candidates": 32}

    @app.post("/api/bench/preview")
    async def bench_preview(payload: dict = Body(...)) -> dict[str, Any]:
        """Expand a sweep without running it, so the UI can show the cost."""
        try:
            model_path = payload.get("model_path", "")
            candidates = build_candidates(payload.get("sweep") or {},
                                          cfg.params_for(model_path), model_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"candidates": candidates, "count": len(candidates)}

    # ------------------------------------------------------------- processes

    @app.get("/api/processes")
    async def processes() -> dict[str, Any]:
        return {"external": find_external_servers(exclude_pids=pool.pids),
                "ours": sorted(pool.pids)}

    @app.post("/api/processes/kill")
    async def kill_process(payload: dict = Body(...)) -> dict[str, Any]:
        """Kill a llama-server this controller does not manage."""
        try:
            pid = int(payload.get("pid"))
        except (TypeError, ValueError):
            raise HTTPException(400, "pid is required") from None
        if pid in pool.pids:
            raise HTTPException(409, "that process is one of ours — use stop instead")
        target = next((p for p in find_external_servers(exclude_pids=pool.pids)
                       if p["pid"] == pid), None)
        if target is None:
            raise HTTPException(404, f"no external llama-server with pid {pid}")

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError as exc:
                raise HTTPException(403, f"not allowed to signal pid {pid} "
                                         f"(different user?)") from exc
            for _ in range(20):
                await asyncio.sleep(0.25)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pool.log(f"killed external llama-server pid {pid}")
                    return {"killed": pid, "signal": sig.name}
        raise HTTPException(500, f"pid {pid} is still alive after SIGKILL")

    # ------------------------------------------------------- huggingface hub

    @app.get("/api/hub/search")
    async def hub_search(q: str, limit: int = 20) -> dict[str, Any]:
        if not q.strip():
            return {"results": []}
        try:
            return {"results": await hub.search(q.strip(), limit)}
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"hugging face unreachable: {exc}") from exc

    @app.get("/api/hub/files")
    async def hub_files(repo: str) -> dict[str, Any]:
        try:
            files = await hub.list_files(repo)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code, f"repo not found: {repo}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"hugging face unreachable: {exc}") from exc
        return {"repo": repo, "files": files}

    @app.post("/api/hub/download")
    async def hub_download(payload: dict = Body(...)) -> dict[str, Any]:
        repo, path = payload.get("repo"), payload.get("path")
        if not repo or not path:
            raise HTTPException(400, "repo and path are required")
        try:
            job = downloader.start(repo, path)
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.public()

    @app.get("/api/hub/downloads")
    async def hub_downloads() -> dict[str, Any]:
        return {"downloads": downloader.public(), "dest": str(downloader.dest_dir)}

    @app.post("/api/hub/cancel")
    async def hub_cancel(payload: dict = Body(...)) -> dict[str, Any]:
        job_id = payload.get("id")
        if not downloader.cancel(int(job_id)) and not downloader.forget(int(job_id)):
            raise HTTPException(404, "no such download")
        return {"ok": True}

    # --------------------------------------------------------------- servers

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {"instances": pool.status()}

    @app.post("/api/server/start")
    async def start(payload: dict = Body(...)) -> dict[str, Any]:
        params = LaunchParams.from_dict(payload.get("params") or {})
        model_path = payload.get("model_path") or ""
        if params.mode != "router" and not model_path:
            raise HTTPException(400, "model_path is required")
        try:
            result = await pool.start(model_path, params, force=bool(payload.get("force")))
        except MemoryError as exc:
            # 507 so the UI can offer "start anyway" instead of just failing
            raise HTTPException(507, str(exc)) from exc
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

        if params.mode == "router":
            cfg.defaults.models_dir = params.models_dir
        else:
            cfg.presets[model_path] = asdict(params)
            cfg.last_model = model_path
        cfg.save()
        return result

    @app.post("/api/server/stop")
    async def stop(payload: dict = Body(default={})) -> dict[str, Any]:
        instance = instance_or_404(payload.get("id"))
        await instance.stop()
        return instance.status()

    @app.post("/api/server/restart")
    async def restart(payload: dict = Body(default={})) -> dict[str, Any]:
        instance = instance_or_404(payload.get("id"))
        return await pool.restart(instance.id)

    @app.post("/api/server/remove")
    async def remove(payload: dict = Body(default={})) -> dict[str, Any]:
        instance = instance_or_404(payload.get("id"))
        if instance.pid:
            raise HTTPException(409, "instance is still running — stop it first")
        await pool.remove(instance.id)
        return {"ok": True}

    @app.post("/api/server/clear-port")
    async def clear_port(payload: dict = Body(default={})) -> dict[str, Any]:
        port = int(payload.get("port") or cfg.defaults.port)
        host = payload.get("host") or cfg.defaults.host
        try:
            cleared = await pool.clear_port(host, port)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"cleared": cleared, "port": port}

    # ------------------------------------------------------------- autostart

    @app.get("/api/autostart")
    async def get_autostart() -> dict[str, Any]:
        return {"entries": cfg.autostart}

    @app.post("/api/autostart")
    async def set_autostart(payload: dict = Body(default={})) -> dict[str, Any]:
        """Save the servers to bring up on boot — by default whatever runs now."""
        if "entries" in payload:
            cfg.autostart = payload["entries"] or []
        else:
            cfg.autostart = [
                {"model_path": instance.model_path, "params": asdict(instance.params)}
                for instance in sorted(pool.instances.values(), key=lambda i: i.id)
                if instance.pid
            ]
        cfg.save()
        return {"entries": cfg.autostart}

    # ---------------------------------------------------------------- router

    # declared before the {action} route below, which would otherwise swallow it
    @app.post("/api/router/refresh")
    async def router_refresh(payload: dict = Body(...)) -> dict[str, Any]:
        instance = instance_or_404(payload.get("id"))
        await instance.refresh_router_models()
        return {"models": instance.router_models}

    @app.post("/api/router/{action}")
    async def router_action(action: str, payload: dict = Body(...)) -> dict[str, Any]:
        if action not in ("load", "unload"):
            raise HTTPException(404, "unknown router action")
        instance = instance_or_404(payload.get("id"))
        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(400, "model is required")
        try:
            return await instance.router_action(action, model_id)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"router unreachable: {exc}") from exc

    # ------------------------------------------------------------------ logs

    @app.get("/api/logs")
    async def logs(limit: int = 500, instance: int | None = None) -> dict[str, Any]:
        return {"logs": pool.recent_logs(limit, instance)}

    @app.get("/api/metrics/history")
    async def history() -> dict[str, Any]:
        return {"history": list(monitor.history)}

    # ------------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        host = ws.client.host if ws.client else ""
        if not allowed(ws.query_params.get("token", ""), host):
            await ws.close(code=1008, reason="authentication required")
            return
        await ws.accept()
        clients.add(ws)
        queue = pool.subscribe()
        try:
            await ws.send_text(json.dumps({
                "type": "hello",
                "data": {
                    "system": monitor.static_info(),
                    "servers": pool.status(),
                    "logs": pool.recent_logs(300),
                    "history": list(monitor.history)[-120:],
                },
            }))
            while True:
                record = await queue.get()
                await ws.send_text(json.dumps({"type": "log", "data": record}))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            clients.discard(ws)
            pool.unsubscribe(queue)

    # ------------------------------------------------------ chat passthrough

    @app.post("/api/chat")
    async def chat(payload: dict = Body(...)) -> Any:
        """Forward a chat completion to one running server, streaming back."""
        instance_id = payload.pop("instance", None)
        instance = instance_or_404(instance_id) if instance_id is not None else None
        if instance is None:
            candidates = pool.ready()
            if not candidates:
                raise HTTPException(409, "no llama-server is ready")
            instance = candidates[0]
        if instance.state.value != "ready":
            raise HTTPException(409, f"{instance.name} is not ready")

        body = {"stream": True, **payload}
        url = f"{instance.base_url}/v1/chat/completions"

        async def relay():
            timeout = httpx.Timeout(10.0, read=600.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream("POST", url, json=body) as response:
                        async for chunk in response.aiter_raw():
                            yield chunk
                except httpx.HTTPError as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.api_route("/proxy/{instance_id}/{path:path}", methods=["GET", "POST"])
    async def proxy(instance_id: int, path: str, request: Request) -> Any:
        """Thin passthrough to a server's own endpoints (/slots, /metrics …)."""
        instance = instance_or_404(instance_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.request(
                    request.method,
                    f"{instance.base_url}/{path}",
                    content=await request.body(),
                    headers={"content-type": request.headers.get("content-type",
                                                                 "application/json")},
                )
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"llama-server unreachable: {exc}") from exc
        return StreamingResponse(iter([response.content]), status_code=response.status_code,
                                 media_type=response.headers.get("content-type", "text/plain"))

    # ------------------------------------------------------------------- web

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        async def root() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app

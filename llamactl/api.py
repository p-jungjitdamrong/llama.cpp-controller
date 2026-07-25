"""HTTP + websocket API and static hosting for the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Config, LaunchParams
from .metrics import Monitor
from .models import ModelIndex
from .supervisor import LlamaSupervisor, find_external_servers

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(cfg: Config) -> FastAPI:
    monitor = Monitor(history_size=cfg.history_size)
    index = ModelIndex()
    supervisor = LlamaSupervisor(cfg)
    state: dict[str, Any] = {"latest": None, "clients": set()}

    async def sampler() -> None:
        """Single 1 Hz sampling loop shared by every connected client."""
        while True:
            try:
                # off the event loop: some GPU backends shell out to a tool
                sample = await asyncio.to_thread(monitor.sample, supervisor.pid)
                sample["server"] = supervisor.status()
                state["latest"] = sample
                payload = json.dumps({"type": "metrics", "data": sample})
                for ws in list(state["clients"]):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        state["clients"].discard(ws)
            except Exception as exc:  # keep the loop alive whatever happens
                print(f"[sampler] {exc!r}")
            await asyncio.sleep(1.0)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(sampler())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await supervisor.shutdown()

    app = FastAPI(title="llama-controller", docs_url="/api/docs", lifespan=lifespan)

    # ------------------------------------------------------------------ meta

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "system": monitor.static_info(),
            "config": cfg.to_dict(),
            "server_bin_exists": cfg.server_bin.is_file(),
            "external_servers": find_external_servers(exclude_pid=supervisor.pid),
            "orphan_on_llama_port": supervisor.find_orphan() if not supervisor.pid else None,
        }

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return cfg.to_dict()

    @app.post("/api/config")
    async def set_config(patch: dict = Body(...)) -> dict[str, Any]:
        for key in ("llama_server_bin", "model_dirs", "history_size", "log_buffer_lines"):
            if key in patch:
                setattr(cfg, key, patch[key])
        if "defaults" in patch:
            cfg.defaults = LaunchParams.from_dict(patch["defaults"])
        cfg.save()
        return cfg.to_dict()

    # ---------------------------------------------------------------- models

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        found = await asyncio.to_thread(index.scan, cfg.resolved_model_dirs())
        for entry in found:
            entry["params"] = asdict(cfg.params_for(entry["path"]))
        return {"models": found, "dirs": [str(d) for d in cfg.resolved_model_dirs()]}

    # ---------------------------------------------------------------- server

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return supervisor.status()

    @app.post("/api/server/start")
    async def start(payload: dict = Body(...)) -> dict[str, Any]:
        model_path = payload.get("model_path")
        if not model_path:
            raise HTTPException(400, "model_path is required")
        params = LaunchParams.from_dict(payload.get("params") or {})
        try:
            result = await supervisor.start(model_path, params)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        cfg.presets[model_path] = asdict(params)
        cfg.last_model = model_path
        cfg.save()
        return result

    @app.post("/api/server/stop")
    async def stop() -> dict[str, Any]:
        return await supervisor.stop()

    @app.post("/api/server/clear-port")
    async def clear_port() -> dict[str, Any]:
        """Kill a llama-server left on our port by a previous controller run."""
        orphan = supervisor.find_orphan()
        if orphan is None:
            return {"cleared": False, "reason": "no orphaned llama-server on that port"}
        host, port = supervisor.bind
        try:
            await supervisor.clear_port(host, port)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"cleared": True, "pid": orphan["pid"]}

    @app.post("/api/server/restart")
    async def restart() -> dict[str, Any]:
        try:
            return await supervisor.restart()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ------------------------------------------------------------------ logs

    @app.get("/api/logs")
    async def logs(limit: int = 500) -> dict[str, Any]:
        entries = list(supervisor.logs)[-limit:]
        return {"logs": entries, "total": len(supervisor.logs)}

    @app.get("/api/metrics/history")
    async def history() -> dict[str, Any]:
        return {"history": list(monitor.history)}

    # ------------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        state["clients"].add(ws)
        queue = supervisor.subscribe()
        try:
            await ws.send_text(json.dumps({
                "type": "hello",
                "data": {
                    "system": monitor.static_info(),
                    "server": supervisor.status(),
                    "logs": list(supervisor.logs)[-300:],
                    "history": list(monitor.history)[-120:],
                },
            }))
            while True:
                record = await queue.get()
                await ws.send_text(json.dumps({"type": "log", "data": record}))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state["clients"].discard(ws)
            supervisor.unsubscribe(queue)

    # ------------------------------------------------------ chat passthrough

    @app.post("/api/chat")
    async def chat(payload: dict = Body(...)) -> Any:
        """Forward a chat completion to the running llama-server, streaming back."""
        if supervisor.state.value != "ready":
            raise HTTPException(409, "llama-server is not ready")
        body = {"stream": True, **payload}
        url = f"{supervisor.base_url}/v1/chat/completions"

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

    @app.api_route("/proxy/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request) -> Any:
        """Thin passthrough to llama-server's own endpoints (/slots, /metrics …)."""
        url = f"{supervisor.base_url}/{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.request(
                    request.method,
                    url,
                    content=await request.body(),
                    headers={"content-type": request.headers.get("content-type", "application/json")},
                )
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"llama-server unreachable: {exc}") from exc
        media = response.headers.get("content-type", "text/plain")
        return StreamingResponse(iter([response.content]), status_code=response.status_code,
                                 media_type=media)

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

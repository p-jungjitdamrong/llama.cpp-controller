"""The agent loop: talk to a model, run the tools it asks for, repeat.

One run is pinned to one llama-server instance for its whole life. That is
deliberate — llama.cpp reuses the KV cache of a shared prefix, and an agent
re-sends the entire conversation on every step, so moving between instances
would throw away the only thing that makes the loop affordable on modest
hardware.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .tools import ToolError, ToolRegistry

SYSTEM_PROMPT = """You are an assistant with access to tools on the user's machine.

Work in small steps: call a tool, read the result, then decide the next move.
Prefer targeted calls — grep and a line range beat reading a whole file.
When you have enough information, answer directly and cite the paths you used.
If a tool fails, read the error and try a different approach rather than repeating
the same call."""


@dataclass
class Budget:
    max_steps: int = 8
    max_seconds: float = 300.0
    max_tokens: int = 800  # per model turn

    @classmethod
    def from_dict(cls, data: dict | None) -> "Budget":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Session:
    id: str
    title: str = ""
    instance_id: int | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "instance_id": self.instance_id,
                "messages": self.messages, "created_at": self.created_at,
                "updated_at": self.updated_at,
                "turns": sum(1 for m in self.messages if m["role"] == "user")}


class AgentRun:
    def __init__(self, run_id: str, session: Session) -> None:
        self.id = run_id
        self.session = session
        self.cancelled = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.started_at = time.time()
        self.steps = 0
        self.last_call: tuple[str, str] | None = None


class AgentManager:
    """Owns sessions and in-flight runs; streams events through `publish`."""

    def __init__(self, registry: ToolRegistry, store_path: Path,
                 publish: Callable[[dict[str, Any]], None]) -> None:
        self.registry = registry
        self.store_path = store_path
        self.publish = publish
        self.sessions: dict[str, Session] = {}
        self.runs: dict[str, AgentRun] = {}
        self._load()

    # ------------------------------------------------------------- sessions

    def _load(self) -> None:
        try:
            raw = json.loads(self.store_path.read_text())
        except (OSError, ValueError):
            return
        for item in raw.get("sessions", []):
            session = Session(id=item["id"], title=item.get("title", ""),
                              instance_id=item.get("instance_id"),
                              messages=item.get("messages", []),
                              created_at=item.get("created_at", time.time()),
                              updated_at=item.get("updated_at", time.time()))
            self.sessions[session.id] = session

    def save(self) -> None:
        newest = sorted(self.sessions.values(), key=lambda s: -s.updated_at)[:50]
        try:
            self.store_path.write_text(json.dumps(
                {"sessions": [s.public() for s in newest]}, indent=2))
        except OSError:
            pass

    def session(self, session_id: str | None) -> Session:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        session = Session(id=uuid.uuid4().hex[:12])
        self.sessions[session.id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        if self.sessions.pop(session_id, None) is None:
            return False
        self.save()
        return True

    # ----------------------------------------------------------------- runs

    def start(self, instance: Any, session_id: str | None, message: str,
              groups: list[str] | None, budget: Budget) -> AgentRun:
        session = self.session(session_id)
        session.instance_id = instance.id
        if not session.title:
            session.title = message.strip()[:60]
        run = AgentRun(uuid.uuid4().hex[:12], session)
        self.runs[run.id] = run
        run.task = asyncio.create_task(
            self._run(run, instance, message, groups or [], budget))
        return run

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None:
            return False
        run.cancelled.set()
        return True

    def _emit(self, run: AgentRun, event: str, **data: Any) -> None:
        self.publish({"run_id": run.id, "session_id": run.session.id,
                      "event": event, "ts": time.time(), **data})

    async def _run(self, run: AgentRun, instance: Any, message: str,
                   groups: list[str], budget: Budget) -> None:
        session = run.session
        if not session.messages:
            session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        session.messages.append({"role": "user", "content": message})
        tools = self.registry.schemas(groups or None)
        deadline = time.time() + budget.max_seconds
        self._emit(run, "start", instance=instance.id, model=instance.name,
                   tools=[t["function"]["name"] for t in tools], title=session.title)

        try:
            timeout = httpx.Timeout(20.0, read=budget.max_seconds + 30)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for step in range(1, budget.max_steps + 1):
                    run.steps = step
                    if run.cancelled.is_set():
                        self._emit(run, "cancelled", step=step)
                        return
                    if time.time() > deadline:
                        self._emit(run, "budget", reason="time limit reached", step=step)
                        return

                    self._emit(run, "step", step=step)
                    content, calls, finish, timings = await self._one_turn(
                        run, client, instance, session.messages, tools, budget)

                    if finish == "error":
                        return
                    if calls:
                        session.messages.append(
                            {"role": "assistant",
                             "content": content.strip() or None,
                             "tool_calls": calls})
                        for call in calls:
                            await self._run_tool(run, session, call)
                        continue

                    session.messages.append({"role": "assistant", "content": content})
                    session.updated_at = time.time()
                    self._emit(run, "final", content=content, step=step,
                               timings=timings,
                               seconds=round(time.time() - run.started_at, 1))
                    return

                self._emit(run, "budget", reason=f"stopped after {budget.max_steps} steps",
                           step=budget.max_steps)
        except httpx.HTTPError as exc:
            self._emit(run, "error", message=f"model unreachable: {exc}")
        except asyncio.CancelledError:
            self._emit(run, "cancelled", step=run.steps)
            raise
        except Exception as exc:  # never let a run take the controller down
            self._emit(run, "error", message=f"{type(exc).__name__}: {exc}")
        finally:
            session.updated_at = time.time()
            self.save()
            self.runs.pop(run.id, None)
            self._emit(run, "done", steps=run.steps,
                       seconds=round(time.time() - run.started_at, 1))

    async def _one_turn(self, run: AgentRun, client: httpx.AsyncClient, instance: Any,
                        messages: list[dict], tools: list[dict], budget: Budget):
        """One model turn, streamed. Returns (content, tool_calls, finish, timings)."""
        payload = {
            "messages": messages,
            "tools": tools,
            "stream": True,
            "max_tokens": budget.max_tokens,
            "temperature": 0.3,
        }
        content = ""
        partial: dict[int, dict[str, Any]] = {}
        finish = "stop"
        timings: dict[str, Any] = {}

        async with client.stream("POST", f"{instance.base_url}/v1/chat/completions",
                                 json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:300]
                self._emit(run, "error", message=f"HTTP {response.status_code}: {body}")
                return "", [], "error", {}
            async for line in response.aiter_lines():
                if run.cancelled.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except ValueError:
                    continue
                if data.get("timings"):
                    timings = data["timings"]
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
                    self._emit(run, "delta", text=delta["content"])
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index", 0)
                    slot = partial.setdefault(
                        index, {"id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        slot["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    if function.get("name"):
                        slot["function"]["name"] = function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        calls = [partial[i] for i in sorted(partial)] if partial else []
        for call in calls:
            call.setdefault("id", uuid.uuid4().hex[:16])
        return content, calls, finish, timings

    async def _run_tool(self, run: AgentRun, session: Session, call: dict) -> None:
        name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or "{}"
        started = time.monotonic()
        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        except ValueError:
            arguments = {}
            output, ok = f"arguments were not valid JSON: {raw_args[:200]}", False
        else:
            self._emit(run, "tool_call", name=name, arguments=arguments)
            signature = (name, json.dumps(arguments, sort_keys=True))
            if signature == run.last_call:
                # a repeat costs a whole step, and on slow hardware a step is
                # ten seconds the user is watching — push the model along instead
                output, ok = ("This is identical to your previous call and the "
                              "result has not changed. Use what you already have, "
                              "or try a different path or tool."), True
            else:
                run.last_call = signature
                try:
                    output = await asyncio.to_thread(self.registry.call, name, arguments)
                    ok = True
                except ToolError as exc:
                    output, ok = f"error: {exc}", False

        elapsed = round(time.monotonic() - started, 2)
        self._emit(run, "tool_result", name=name, ok=ok, seconds=elapsed,
                   output=output[:2000])
        session.messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "name": name,
            "content": output,
        })

"""Tools the agent can call, and the sandbox around them.

Phase 1 is read-only: nothing here writes, deletes or executes. Every path is
resolved and checked against an allowlist of roots and a denylist of things that
should never end up in a model's context (keys, credentials, wallets).

Tool results are truncated hard. With an 8k context an unbounded `grep` result
would eat the whole conversation in one step.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_RESULT_CHARS = 4000

# never readable, whatever the roots say
DENY_PARTS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", "gcloud", ".password-store"}
DENY_GLOBS = ("*.pem", "*.key", "id_rsa*", "id_ed25519*", "*.p12", "*.pfx",
              ".env", ".env.*", "*.kdbx", "credentials", "*.sqlite-wal")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache",
             "dist", "build", ".mypy_cache", ".pytest_cache"}


class ToolError(RuntimeError):
    """Raised for anything the model should see as a failed call, not a crash."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    mutating: bool = False
    # "local" marks output that must not travel to a cloud provider
    sensitivity: str = "local"
    group: str = "general"

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}


class Sandbox:
    """Path allowlist. Everything the file tools touch goes through here."""

    def __init__(self, roots: list[str]) -> None:
        self.roots = [Path(r).expanduser().resolve() for r in roots]

    def describe(self) -> list[str]:
        return [str(r) for r in self.roots]

    def check(self, raw: str, must_exist: bool = True) -> Path:
        if not raw or not str(raw).strip():
            raise ToolError("path is required")
        path = Path(str(raw)).expanduser()
        try:
            resolved = (path if path.is_absolute() else self.roots[0] / path).resolve()
        except OSError as exc:
            raise ToolError(f"cannot resolve path: {exc}") from exc

        if not any(resolved == root or root in resolved.parents for root in self.roots):
            raise ToolError(
                f"{resolved} is outside the allowed directories "
                f"({', '.join(self.describe())})")
        parts = set(resolved.parts)
        if parts & DENY_PARTS:
            raise ToolError("that location holds credentials and is not readable")
        if any(fnmatch.fnmatch(resolved.name, pattern) for pattern in DENY_GLOBS):
            raise ToolError("that file looks like a secret and is not readable")
        if must_exist and not resolved.exists():
            raise ToolError(f"no such file or directory: {resolved}")
        return resolved


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… truncated, {len(text) - limit} more characters"


def _is_texty(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" not in f.read(2048)
    except OSError:
        return False


def build_file_tools(sandbox: Sandbox) -> list[Tool]:
    def list_dir(path: str = ".", max_entries: int = 100) -> str:
        target = sandbox.check(path)
        if not target.is_dir():
            raise ToolError(f"{target} is not a directory")
        rows = []
        for entry in sorted(target.iterdir())[: int(max_entries)]:
            if entry.name in SKIP_DIRS:
                continue
            try:
                if entry.is_dir():
                    rows.append(f"{entry.name}/")
                else:
                    rows.append(f"{entry.name}  {entry.stat().st_size}")
            except OSError:
                continue
        return _truncate(f"{target}\n" + "\n".join(rows) if rows else f"{target} is empty")

    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = sandbox.check(path)
        if not target.is_file():
            raise ToolError(f"{target} is not a file")
        if not _is_texty(target):
            raise ToolError(f"{target} looks binary")
        start, end = max(1, int(start_line)), int(end_line)
        if end < start:
            raise ToolError("end_line must be >= start_line")
        out = []
        with target.open(errors="replace") as f:
            for number, line in enumerate(f, 1):
                if number < start:
                    continue
                if number > end:
                    out.append(f"… file continues past line {end}")
                    break
                out.append(f"{number}\t{line.rstrip()}")
        return _truncate("\n".join(out) or "(empty)")

    def grep(pattern: str, path: str = ".", max_results: int = 40,
             glob: str = "*") -> str:
        root = sandbox.check(path)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad regular expression: {exc}") from exc
        files = [root] if root.is_file() else [
            p for p in root.rglob(glob)
            if p.is_file() and not (set(p.parts) & SKIP_DIRS)
        ]
        hits: list[str] = []
        for file in files[:2000]:
            if len(hits) >= int(max_results):
                break
            if not _is_texty(file):
                continue
            try:
                with file.open(errors="replace") as f:
                    for number, line in enumerate(f, 1):
                        if regex.search(line):
                            hits.append(f"{file}:{number}: {line.strip()[:200]}")
                            if len(hits) >= int(max_results):
                                break
            except OSError:
                continue
        return _truncate("\n".join(hits) or f"no match for {pattern!r} under {root}")

    def find_files(glob: str, path: str = ".", max_results: int = 60) -> str:
        root = sandbox.check(path)
        found = [str(p) for p in root.rglob(glob)
                 if p.is_file() and not (set(p.parts) & SKIP_DIRS)][: int(max_results)]
        return _truncate("\n".join(found) or f"nothing matching {glob} under {root}")

    return [
        Tool("list_dir", "List the entries of a directory.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "directory to list"},
                 "max_entries": {"type": "integer"}},
              "required": ["path"]},
             list_dir, group="files"),
        Tool("read_file",
             "Read a text file. Always prefer a line range over the whole file.",
             {"type": "object", "properties": {
                 "path": {"type": "string"},
                 "start_line": {"type": "integer", "description": "1-based, default 1"},
                 "end_line": {"type": "integer", "description": "default 200"}},
              "required": ["path"]},
             read_file, group="files"),
        Tool("grep", "Search file contents with a regular expression.",
             {"type": "object", "properties": {
                 "pattern": {"type": "string"},
                 "path": {"type": "string", "description": "file or directory to search"},
                 "glob": {"type": "string", "description": "limit to matching names, e.g. *.py"},
                 "max_results": {"type": "integer"}},
              "required": ["pattern"]},
             grep, group="files"),
        Tool("find_files", "Find files by name pattern, e.g. *.md",
             {"type": "object", "properties": {
                 "glob": {"type": "string"},
                 "path": {"type": "string"},
                 "max_results": {"type": "integer"}},
              "required": ["glob"]},
             find_files, group="files"),
    ]


def build_git_tools(sandbox: Sandbox) -> list[Tool]:
    def _git(repo: str, *args: str) -> str:
        root = sandbox.check(repo)
        if not (root / ".git").exists():
            raise ToolError(f"{root} is not a git repository")
        try:
            proc = subprocess.run(["git", "-C", str(root), *args],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"git failed: {exc}") from exc
        if proc.returncode != 0:
            raise ToolError(proc.stderr.strip()[:400] or "git returned an error")
        return _truncate(proc.stdout.strip() or "(no output)")

    return [
        Tool("git_status", "Working tree status of a repository.",
             {"type": "object", "properties": {"repo": {"type": "string"}},
              "required": ["repo"]},
             lambda repo: _git(repo, "status", "--short", "--branch"), group="git"),
        Tool("git_log", "Recent commits, newest first.",
             {"type": "object", "properties": {
                 "repo": {"type": "string"},
                 "count": {"type": "integer", "description": "default 10"}},
              "required": ["repo"]},
             lambda repo, count=10: _git(repo, "log", f"-{int(count)}",
                                         "--pretty=format:%h %ad %an: %s",
                                         "--date=short"), group="git"),
        Tool("git_diff", "Uncommitted changes, or the diff of one path.",
             {"type": "object", "properties": {
                 "repo": {"type": "string"},
                 "path": {"type": "string"}},
              "required": ["repo"]},
             lambda repo, path=None: _git(repo, "diff", "--stat", *( [path] if path else [])),
             group="git"),
    ]


class ToolRegistry:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.tools: dict[str, Tool] = {}
        for tool in build_file_tools(sandbox) + build_git_tools(sandbox):
            self.tools[tool.name] = tool

    @property
    def groups(self) -> list[str]:
        return sorted({t.group for t in self.tools.values()})

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "group": t.group, "mutating": t.mutating,
                 "description": t.description} for t in self.tools.values()]

    def schemas(self, groups: list[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(groups) if groups else None
        return [t.schema() for t in self.tools.values()
                if wanted is None or t.group in wanted]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"no such tool: {name}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        try:
            return tool.handler(**arguments)
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"bad arguments for {name}: {exc}") from exc
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

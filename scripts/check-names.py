#!/usr/bin/env python3
"""Catch names that are used but never defined, without installing a linter.

Python binds names at call time, so a helper lost in a refactor — or a variable
renamed in one place and not another — compiles cleanly and fails only when a
user reaches that line. Both happened here. This walks the AST of each module,
collects everything that could legitimately provide a name, and reports loads
that nothing accounts for.

It is deliberately conservative: it reports only names that appear nowhere as an
import, assignment, def, class, argument, comprehension target, except handler,
global, or builtin. That misses some real faults but never invents one.

Usage: python3 scripts/check-names.py llamactl scripts
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Panuwat Jungjitdamrong

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}


class ModuleScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used: list[tuple[str, int]] = []

    # -- everything that can introduce a name ------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        self._arguments(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._arguments(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.defined.update(node.names)
        self.generic_visit(node)

    visit_Nonlocal = visit_Global  # type: ignore[assignment]

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.defined.add(node.id)
        else:
            self.used.append((node.id, node.lineno))
        self.generic_visit(node)

    def _arguments(self, args: ast.arguments) -> None:
        for group in (args.posonlyargs, args.args, args.kwonlyargs):
            for arg in group:
                self.defined.add(arg.arg)
        for extra in (args.vararg, args.kwarg):
            if extra:
                self.defined.add(extra.arg)


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]
    scan = ModuleScan()
    scan.visit(tree)
    seen: set[str] = set()
    problems = []
    for name, line in scan.used:
        if name in scan.defined or name in BUILTINS or name in seen:
            continue
        seen.add(name)
        problems.append(f"{path}:{line}: '{name}' is used but never defined here")
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path("llamactl")]
    files = sorted(f for root in roots
                   for f in ([root] if root.is_file() else root.rglob("*.py")))
    problems = [problem for f in files for problem in check(f)]
    for problem in problems:
        print(problem)
    print(f"\n{len(files)} files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

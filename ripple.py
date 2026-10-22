"""
Ripple: Open-Source AI Change-Impact Explorer (Hackathon MVP)
=============================================================
Static Python AST call-graph analyzer, NetworkX reverse-reachability engine,
configurable Gemma 4 explanation provider (OpenRouter & Google AI compatible),
and Streamlit UI.

SECURITY & ANALYSIS CONSTRAINTS:
- Never executes code from analyzed repositories (uses ast.parse only).
- Never fabricates graph edges or claims complete coverage; reports unresolved calls.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]


DEFAULT_GEMMA_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GEMMA_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_GEMMA_API_KEY = (
    ""
)


def get_configured_api_key() -> str:
    """Return the active Gemma API key from environment variables or default config."""
    return (
        os.environ.get("GEMMA_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or DEFAULT_GEMMA_API_KEY
    ).strip()


# ============================================================================
# MILESTONE 1: WORKSPACE & ENVIRONMENT INSPECTOR
# ============================================================================

@dataclass(frozen=True)
class EnvironmentReport:
    """Result of inspecting the Python runtime, workspace, and required packages."""

    python_version: str
    python_ok: bool
    workspace_path: str
    workspace_files: List[str]
    dependencies: Dict[str, bool]
    gemma_key_configured: bool
    gemma_api_base: str
    gemma_model: str

    @property
    def all_core_dependencies_installed(self) -> bool:
        return (
            self.python_ok
            and self.dependencies.get("ast", False)
            and self.dependencies.get("networkx", False)
            and self.dependencies.get("pytest", False)
            and self.dependencies.get("streamlit", False)
        )


def inspect_workspace(workspace_dir: Optional[Path] = None) -> EnvironmentReport:
    """Inspect current workspace directory, Python version, and installed packages."""
    root = (workspace_dir or Path.cwd()).resolve()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = (sys.version_info.major, sys.version_info.minor) >= (3, 9)

    required_modules = ["ast", "networkx", "pytest", "streamlit"]
    dep_status: Dict[str, bool] = {}
    for mod_name in required_modules:
        dep_status[mod_name] = importlib.util.find_spec(mod_name) is not None

    discovered_files: List[str] = []
    if root.exists() and root.is_dir():
        for item in sorted(root.iterdir()):
            if not item.name.startswith("."):
                discovered_files.append(item.name + ("/" if item.is_dir() else ""))

    has_key = bool(get_configured_api_key())
    api_base = os.environ.get("GEMMA_API_BASE", DEFAULT_GEMMA_API_BASE)
    model_name = os.environ.get("GEMMA_MODEL", DEFAULT_GEMMA_MODEL)

    return EnvironmentReport(
        python_version=py_ver,
        python_ok=py_ok,
        workspace_path=str(root),
        workspace_files=discovered_files[:50],
        dependencies=dep_status,
        gemma_key_configured=has_key,
        gemma_api_base=api_base,
        gemma_model=model_name,
    )


# ============================================================================
# MILESTONE 8 (FOUNDATION): SAFE REPOSITORY INGESTION & LIMITS
# ============================================================================

@dataclass(frozen=True)
class RepoLimits:
    """Safety guardrails for static repository ingestion."""

    max_files: int = 200
    max_file_bytes: int = 256 * 1024  # 256 KB per source file
    max_total_bytes: int = 5 * 1024 * 1024  # 5 MB across repository
    excluded_dirs: Tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
    )


@dataclass
class IngestedFile:
    """A safely read Python file ready for AST parsing."""

    rel_path: str
    module_name: str
    source: str
    size_bytes: int


@dataclass
class IngestionReport:
    """Summary of safe repository ingestion, including skipped or erred files."""

    root_path: str
    files: List[IngestedFile] = field(default_factory=list)
    skipped_files: List[Tuple[str, str]] = field(default_factory=list)
    total_bytes: int = 0


def rel_path_to_module(rel_path: str) -> str:
    """Convert a relative POSIX path like 'pkg/sub/mod.py' into 'pkg.sub.mod'."""
    clean = rel_path.replace("\\", "/")
    if clean.endswith("/__init__.py"):
        clean = clean[: -len("/__init__.py")]
    elif clean.endswith(".py"):
        clean = clean[: -len(".py")]
    return clean.replace("/", ".").strip(".") or "__root__"


def ingest_repository(
    repo_root: Path, limits: Optional[RepoLimits] = None
) -> IngestionReport:
    """
    Safely discover and read Python files without executing any repository code.
    Rejects symlinks, files outside repo_root, oversized files, and binary payloads.
    """
    active_limits = limits or RepoLimits()
    resolved_root = repo_root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"Repository path is not a valid directory: {repo_root}")

    report = IngestionReport(root_path=str(resolved_root))
    excluded_set = set(active_limits.excluded_dirs)

    candidate_paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in excluded_set and not d.startswith(".")
        )
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                candidate_paths.append(Path(dirpath) / fname)

    for file_path in candidate_paths:
        rel_str = str(file_path.relative_to(resolved_root)).replace("\\", "/")

        if file_path.is_symlink():
            report.skipped_files.append((rel_str, "Skipped symlink for safety"))
            continue

        try:
            resolved_file = file_path.resolve()
            resolved_file.relative_to(resolved_root)
        except ValueError:
            report.skipped_files.append((rel_str, "Path traversal outside root"))
            continue

        if len(report.files) >= active_limits.max_files:
            report.skipped_files.append(
                (rel_str, f"Exceeded max_files limit ({active_limits.max_files})")
            )
            continue

        try:
            stat_info = resolved_file.stat()
        except OSError as exc:
            report.skipped_files.append((rel_str, f"OS stat error: {exc}"))
            continue

        if stat_info.st_size > active_limits.max_file_bytes:
            report.skipped_files.append(
                (
                    rel_str,
                    f"File size {stat_info.st_size}B exceeds max_file_bytes ({active_limits.max_file_bytes}B)",
                )
            )
            continue

        if report.total_bytes + stat_info.st_size > active_limits.max_total_bytes:
            report.skipped_files.append(
                (
                    rel_str,
                    f"Total ingested bytes would exceed max_total_bytes ({active_limits.max_total_bytes}B)",
                )
            )
            continue

        try:
            source_text = resolved_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            report.skipped_files.append((rel_str, f"Read/encoding error: {exc}"))
            continue

        report.files.append(
            IngestedFile(
                rel_path=rel_str,
                module_name=rel_path_to_module(rel_str),
                source=source_text,
                size_bytes=stat_info.st_size,
            )
        )
        report.total_bytes += stat_info.st_size

    return report


# ============================================================================
# MILESTONE 2: STATIC PYTHON AST PARSER
# ============================================================================

@dataclass(frozen=True)
class SymbolNode:
    """Represents a statically discovered function, method, or class symbol."""

    symbol_id: str  # e.g., "billing.tax.calculate_tax" or "billing.invoice.Invoice.total"
    name: str
    kind: str  # "function" | "method" | "class"
    module: str
    file_path: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    is_test: bool = False
    source_excerpt: str = ""


@dataclass(frozen=True)
class ImportBinding:
    """Maps a local alias inside a module to an imported target module or symbol."""

    module: str
    file_path: str
    local_name: str
    target_path: str  # e.g., "billing.tax.calculate_tax" or "billing.tax"
    line: int


@dataclass(frozen=True)
class RawCallSite:
    """A straightforward call discovered inside a function or method body."""

    caller_id: str
    callee_expr: str  # e.g., "calculate_tax", "tax.calculate_tax", "self.compute"
    file_path: str
    line: int
    snippet: str


@dataclass(frozen=True)
class UnresolvedCall:
    """A call site that cannot be deterministically resolved by static analysis."""

    caller_id: str
    file_path: str
    line: int
    expr_description: str
    reason: str


@dataclass
class ParsedRepository:
    """Complete output of Milestone 2 static AST extraction across a repository."""

    symbols: Dict[str, SymbolNode] = field(default_factory=dict)
    imports: List[ImportBinding] = field(default_factory=list)
    raw_calls: List[RawCallSite] = field(default_factory=list)
    unresolved_calls: List[UnresolvedCall] = field(default_factory=list)
    syntax_errors: List[Tuple[str, str]] = field(default_factory=list)


class _ModuleASTVisitor(ast.NodeVisitor):
    """Visits a single Python AST module to extract symbols, imports, and calls."""

    def __init__(
        self, rel_path: str, module_name: str, source_lines: List[str]
    ) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_lines = source_lines

        self.symbols: Dict[str, SymbolNode] = {}
        self.imports: List[ImportBinding] = []
        self.raw_calls: List[RawCallSite] = []
        self.unresolved_calls: List[UnresolvedCall] = []

        self._class_stack: List[str] = []
        self._func_stack: List[str] = []

    def _extract_excerpt(self, start_line: int, end_line: int, max_lines: int = 25) -> str:
        """Return bounded source lines (1-indexed) with line numbers."""
        s_idx = max(0, start_line - 1)
        e_idx = min(len(self.source_lines), s_idx + max_lines, end_line)
        numbered = [
            f"{idx + 1:4d} | {self.source_lines[idx]}"
            for idx in range(s_idx, e_idx)
        ]
        if end_line > e_idx:
            numbered.append(f"... ({end_line - e_idx} more lines truncated)")
        return "\n".join(numbered)

    def _get_line_snippet(self, lineno: int) -> str:
        """Return trimmed single-line source snippet for a 1-indexed line number."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def _is_test_symbol(self, name: str) -> bool:
        base = Path(self.rel_path).name
        return (
            name.startswith("test_")
            or name.startswith("Test")
            or base.startswith("test_")
            or base.endswith("_test.py")
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            target = alias.name if alias.asname else alias.name.split(".")[0]
            self.imports.append(
                ImportBinding(
                    module=self.module_name,
                    file_path=self.rel_path,
                    local_name=local_name,
                    target_path=target,
                    line=getattr(node, "lineno", 1),
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base_mod = node.module or ""
        if node.level > 0:
            parts = self.module_name.split(".")
            pkg_parts = parts[:-1]
            up = max(0, len(pkg_parts) - (node.level - 1))
            prefix = ".".join(pkg_parts[:up])
            base_mod = f"{prefix}.{base_mod}".strip(".") if base_mod else prefix

        for alias in node.names:
            if alias.name == "*":
                self.unresolved_calls.append(
                    UnresolvedCall(
                        caller_id=self.module_name,
                        file_path=self.rel_path,
                        line=getattr(node, "lineno", 1),
                        expr_description=f"from {base_mod} import *",
                        reason="Wildcard import obscures static symbol bindings",
                    )
                )
                continue
            local_name = alias.asname or alias.name
            target = f"{base_mod}.{alias.name}".strip(".")
            self.imports.append(
                ImportBinding(
                    module=self.module_name,
                    file_path=self.rel_path,
                    local_name=local_name,
                    target_path=target,
                    line=getattr(node, "lineno", 1),
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_qual = ".".join([self.module_name, *self._class_stack, node.name])
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        self.symbols[class_qual] = SymbolNode(
            symbol_id=class_qual,
            name=node.name,
            kind="class",
            module=self.module_name,
            file_path=self.rel_path,
            start_line=start_line,
            end_line=end_line,
            parent_class=self._class_stack[-1] if self._class_stack else None,
            is_test=self._is_test_symbol(node.name),
            source_excerpt=self._extract_excerpt(start_line, end_line),
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_func(self, node: ast.AST, name: str) -> None:
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        parent_cls = self._class_stack[-1] if self._class_stack else None
        kind = "method" if parent_cls else "function"

        qual_parts = [self.module_name, *self._class_stack, name]
        symbol_id = ".".join(qual_parts)

        self.symbols[symbol_id] = SymbolNode(
            symbol_id=symbol_id,
            name=name,
            kind=kind,
            module=self.module_name,
            file_path=self.rel_path,
            start_line=start_line,
            end_line=end_line,
            parent_class=parent_cls,
            is_test=self._is_test_symbol(name),
            source_excerpt=self._extract_excerpt(start_line, end_line),
        )

        self._func_stack.append(symbol_id)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        caller_id = self._func_stack[-1] if self._func_stack else self.module_name
        lineno = getattr(node, "lineno", 1)
        snippet = self._get_line_snippet(lineno)

        callee_name = self._extract_call_target(node.func)
        if callee_name is not None:
            if callee_name in ("getattr", "eval", "exec", "__import__"):
                self.unresolved_calls.append(
                    UnresolvedCall(
                        caller_id=caller_id,
                        file_path=self.rel_path,
                        line=lineno,
                        expr_description=snippet or callee_name,
                        reason=f"Dynamic invocation primitive '{callee_name}' cannot be statically resolved",
                    )
                )
            else:
                self.raw_calls.append(
                    RawCallSite(
                        caller_id=caller_id,
                        callee_expr=callee_name,
                        file_path=self.rel_path,
                        line=lineno,
                        snippet=snippet,
                    )
                )
        else:
            expr_desc = snippet or ast.dump(node.func)
            self.unresolved_calls.append(
                UnresolvedCall(
                    caller_id=caller_id,
                    file_path=self.rel_path,
                    line=lineno,
                    expr_description=expr_desc,
                    reason="Non-trivial or chained call expression outside supported static subset",
                )
            )

        self.generic_visit(node)

    @staticmethod
    def _extract_call_target(func_node: ast.AST) -> Optional[str]:
        """
        Support straightforward calls:
        - direct name: foo() -> 'foo'
        - dotted attribute: mod.foo() or self.bar() -> 'mod.foo' / 'self.bar'
        Returns None for dynamic expressions like a().b(), callbacks[i](), etc.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            parts: List[str] = [func_node.attr]
            curr: ast.AST = func_node.value
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
                return ".".join(reversed(parts))
        return None


def parse_ingested_repository(ingestion: IngestionReport) -> ParsedRepository:
    """Parse all safely ingested files using Python's built-in ast module."""
    parsed = ParsedRepository()

    for item in ingestion.files:
        try:
            tree = ast.parse(item.source, filename=item.rel_path)
        except SyntaxError as exc:
            parsed.syntax_errors.append(
                (item.rel_path, f"Line {exc.lineno}: {exc.msg}")
            )
            continue

        lines = item.source.splitlines()
        visitor = _ModuleASTVisitor(item.rel_path, item.module_name, lines)
        visitor.visit(tree)

        parsed.symbols.update(visitor.symbols)
        parsed.imports.extend(visitor.imports)
        parsed.raw_calls.extend(visitor.raw_calls)
        parsed.unresolved_calls.extend(visitor.unresolved_calls)

    return parsed


# ============================================================================
# MILESTONE 3: DIRECTED DEPENDENCY GRAPH (NETWORKX)
# ============================================================================

_PYTHON_BUILTINS: Set[str] = {
    "abs", "all", "any", "bool", "dict", "enumerate", "Exception", "filter",
    "float", "frozenset", "hasattr", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "object", "open",
    "ord", "pow", "print", "property", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
    "type", "ValueError", "TypeError", "KeyError", "RuntimeError", "zip",
}


@dataclass(frozen=True)
class GraphEdgeRecord:
    """Explicit directed dependency edge: caller_id -> callee_id."""

    caller_id: str
    callee_id: str
    relationship: str  # Always "CALLS" for call edges
    resolution_kind: str  # "SAME_MODULE" | "IMPORT_RESOLVED" | "SELF_METHOD" | "UNIQUE_NAME_MATCH"
    file_path: str
    line: int
    snippet: str


@dataclass
class DependencyGraphBundle:
    """Wraps the NetworkX DiGraph alongside resolved edges and unresolved calls."""

    graph: Any  # nx.DiGraph
    parsed: ParsedRepository
    edges: List[GraphEdgeRecord] = field(default_factory=list)
    unresolved_calls: List[UnresolvedCall] = field(default_factory=list)


def build_dependency_graph(parsed: ParsedRepository) -> DependencyGraphBundle:
    """
    Build a directed NetworkX graph from parsed symbols, imports, and call sites.

    EDGE DIRECTION CONVENTION:
        caller_id -> callee_id  (with edge attribute relationship="CALLS")
    Meaning: if function A calls function B, there is a directed edge A -> B.
    When B changes, reverse reachability traverses incoming edges (predecessors of B).
    """
    if nx is None:
        raise RuntimeError("networkx is required to build the dependency graph.")

    digraph = nx.DiGraph()

    for sym_id, sym in sorted(parsed.symbols.items()):
        digraph.add_node(
            sym_id,
            name=sym.name,
            kind=sym.kind,
            module=sym.module,
            file_path=sym.file_path,
            start_line=sym.start_line,
            end_line=sym.end_line,
            is_test=sym.is_test,
        )

    module_imports: Dict[str, Dict[str, str]] = {}
    for imp in parsed.imports:
        module_imports.setdefault(imp.module, {})[imp.local_name] = imp.target_path

    symbols_by_short_name: Dict[str, List[SymbolNode]] = {}
    for sym in parsed.symbols.values():
        symbols_by_short_name.setdefault(sym.name, []).append(sym)

    resolved_edges: List[GraphEdgeRecord] = []
    all_unresolved: List[UnresolvedCall] = list(parsed.unresolved_calls)

    for call in parsed.raw_calls:
        caller_sym = parsed.symbols.get(call.caller_id)
        caller_module = caller_sym.module if caller_sym else call.caller_id
        caller_class = caller_sym.parent_class if caller_sym else None

        expr = call.callee_expr
        if expr in _PYTHON_BUILTINS:
            continue

        target_id: Optional[str] = None
        resolution_kind = ""

        # Case A: self.method() inside a class method
        if expr.startswith("self.") and caller_class:
            method_name = expr.split(".", 1)[1]
            candidate = f"{caller_module}.{caller_class}.{method_name}"
            if candidate in parsed.symbols:
                target_id = candidate
                resolution_kind = "SELF_METHOD"

        # Case B: Direct symbol inside the same module
        if target_id is None and "." not in expr:
            same_mod_candidate = f"{caller_module}.{expr}"
            if same_mod_candidate in parsed.symbols:
                target_id = same_mod_candidate
                resolution_kind = "SAME_MODULE"

        # Case C: Direct imported symbol or module.symbol via import alias
        if target_id is None:
            imp_map = module_imports.get(caller_module, {})
            parts = expr.split(".")
            head = parts[0]
            tail = ".".join(parts[1:])
            if head in imp_map:
                mapped_base = imp_map[head]
                candidate = f"{mapped_base}.{tail}".strip(".") if tail else mapped_base
                if candidate in parsed.symbols:
                    target_id = candidate
                    resolution_kind = "IMPORT_RESOLVED"
                elif f"{candidate}.__init__" in parsed.symbols:
                    target_id = f"{candidate}.__init__"
                    resolution_kind = "IMPORT_RESOLVED"

        # Case D: Class instantiation where ClassName.__init__ exists
        if target_id in parsed.symbols and parsed.symbols[target_id].kind == "class":
            init_candidate = f"{target_id}.__init__"
            if init_candidate in parsed.symbols:
                target_id = init_candidate

        # Case E: Unambiguous single function match in repository
        if target_id is None and "." not in expr:
            matches = [
                s for s in symbols_by_short_name.get(expr, []) if s.kind == "function"
            ]
            if len(matches) == 1:
                target_id = matches[0].symbol_id
                resolution_kind = "UNIQUE_NAME_MATCH"

        if target_id is not None and target_id in parsed.symbols:
            if call.caller_id not in digraph:
                digraph.add_node(
                    call.caller_id,
                    name=call.caller_id.split(".")[-1],
                    kind="module",
                    module=caller_module,
                    file_path=call.file_path,
                    start_line=1,
                    end_line=1,
                    is_test=False,
                )
            edge_rec = GraphEdgeRecord(
                caller_id=call.caller_id,
                callee_id=target_id,
                relationship="CALLS",
                resolution_kind=resolution_kind,
                file_path=call.file_path,
                line=call.line,
                snippet=call.snippet,
            )
            resolved_edges.append(edge_rec)
            digraph.add_edge(
                call.caller_id,
                target_id,
                relationship="CALLS",
                resolution_kind=resolution_kind,
                file_path=call.file_path,
                line=call.line,
                snippet=call.snippet,
            )
        else:
            all_unresolved.append(
                UnresolvedCall(
                    caller_id=call.caller_id,
                    file_path=call.file_path,
                    line=call.line,
                    expr_description=call.snippet or f"{expr}()",
                    reason=f"Target '{expr}' could not be matched to a repository definition",
                )
            )

    return DependencyGraphBundle(
        graph=digraph,
        parsed=parsed,
        edges=resolved_edges,
        unresolved_calls=all_unresolved,
    )


# ============================================================================
# MILESTONE 4: REVERSE REACHABILITY ENGINE
# ============================================================================

@dataclass(frozen=True)
class ImpactReport:
    """
    Deterministic reverse-reachability result for a selected target function.
    Guarantees that `target_symbol_id` is excluded from `affected_nodes` and `direct_callers`.
    """

    target_symbol_id: str
    direct_callers: List[str]
    affected_nodes: List[str]  # All transitive callers, EXCLUDING target_symbol_id
    candidate_tests: List[str]
    impact_paths: List[List[str]]  # Paths ordered [caller, ..., target_symbol_id]
    supporting_edges: List[GraphEdgeRecord]


def compute_reverse_reachability(
    bundle: DependencyGraphBundle,
    target_symbol_id: str,
    max_depth: int = 15,
) -> ImpactReport:
    """
    Traverse incoming 'CALLS' edges backwards from `target_symbol_id` to find
    statically discoverable direct callers, transitive callers, and candidate tests.
    Strictly excludes `target_symbol_id` from `affected_nodes` (even under recursion).
    """
    g = bundle.graph
    if target_symbol_id not in g:
        raise KeyError(f"Symbol '{target_symbol_id}' not found in dependency graph.")

    direct_callers = sorted(
        pred
        for pred in g.predecessors(target_symbol_id)
        if pred != target_symbol_id
        and g.edges[pred, target_symbol_id].get("relationship") == "CALLS"
    )

    visited_callers: Set[str] = set()
    impact_paths: List[List[str]] = []
    queue: List[Tuple[str, List[str]]] = [(target_symbol_id, [target_symbol_id])]

    while queue:
        current, path_to_target = queue.pop(0)
        if len(path_to_target) - 1 >= max_depth:
            continue

        for caller in sorted(g.predecessors(current)):
            edge_data = g.edges[caller, current]
            if edge_data.get("relationship") != "CALLS":
                continue
            if caller == target_symbol_id or caller in path_to_target:
                continue

            new_path = [caller] + path_to_target
            impact_paths.append(new_path)

            if caller not in visited_callers:
                visited_callers.add(caller)
                queue.append((caller, new_path))

    visited_callers.discard(target_symbol_id)
    affected_nodes = sorted(visited_callers)

    candidate_tests = sorted(
        node_id
        for node_id in affected_nodes
        if bool(g.nodes[node_id].get("is_test", False))
        or node_id.split(".")[-1].startswith("test_")
    )

    relevant_set = set(affected_nodes) | {target_symbol_id}
    supporting_edges = [
        edge
        for edge in bundle.edges
        if edge.caller_id in relevant_set and edge.callee_id in relevant_set
    ]

    return ImpactReport(
        target_symbol_id=target_symbol_id,
        direct_callers=direct_callers,
        affected_nodes=affected_nodes,
        candidate_tests=candidate_tests,
        impact_paths=impact_paths,
        supporting_edges=supporting_edges,
    )


# ============================================================================
# MILESTONE 7: GEMMA 4 PROVIDER INTERFACE, PROMPT BUILDER & VALIDATOR
# ============================================================================

@dataclass(frozen=True)
class VerifiedFact:
    """A deterministic caller-callee fact grounded in the static AST graph."""

    caller_id: str
    callee_id: str
    evidence_location: str
    summary: str


@dataclass(frozen=True)
class InferredRisk:
    """A model-inferred risk or behavioral hypothesis (explicitly marked as non-deterministic)."""

    affected_symbol: str
    hypothesis: str
    confidence_note: str


@dataclass(frozen=True)
class ValidatedGemmaExplanation:
    """Validated explanation that strictly separates deterministic facts from AI suggestions."""

    provider_name: str
    model_name: str
    target_symbol_id: str
    verified_facts: List[VerifiedFact]
    inferred_risks: List[InferredRisk]
    suggested_test_checks: List[str]
    coverage_disclaimer: str
    validation_warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GemmaPromptPayload:
    """Bounded source excerpts and verified graph relationships sent to Gemma."""

    target_symbol: SymbolNode
    direct_callers: List[str]
    transitive_callers: List[str]
    candidate_tests: List[str]
    verified_edges: List[Dict[str, Any]]
    caller_excerpts: Dict[str, str]
    unresolved_count: int


class InferenceProvider(Protocol):
    """Configurable provider interface for Gemma 4 inference."""

    def generate_explanation(self, payload: GemmaPromptPayload) -> Dict[str, Any]:
        ...


def build_gemma_prompt_payload(
    bundle: DependencyGraphBundle,
    impact: ImpactReport,
    max_caller_excerpts: int = 5,
) -> GemmaPromptPayload:
    """Construct a bounded payload of source evidence and verified graph edges."""
    target_sym = bundle.parsed.symbols[impact.target_symbol_id]

    excerpts: Dict[str, str] = {}
    for caller_id in impact.affected_nodes[:max_caller_excerpts]:
        if caller_id in bundle.parsed.symbols:
            excerpts[caller_id] = bundle.parsed.symbols[caller_id].source_excerpt

    verified_edge_dicts = [
        {
            "caller_id": e.caller_id,
            "callee_id": e.callee_id,
            "file_path": e.file_path,
            "line": e.line,
            "snippet": e.snippet,
        }
        for e in impact.supporting_edges
    ]

    return GemmaPromptPayload(
        target_symbol=target_sym,
        direct_callers=impact.direct_callers,
        transitive_callers=impact.affected_nodes,
        candidate_tests=impact.candidate_tests,
        verified_edges=verified_edge_dicts,
        caller_excerpts=excerpts,
        unresolved_count=len(bundle.unresolved_calls),
    )


def format_gemma_system_prompt(payload: GemmaPromptPayload) -> str:
    """Create the JSON-constrained instruction prompt for Gemma 4."""
    return (
        "You are Ripple's static change-impact analysis assistant powered by Gemma 4.\n"
        "RULES:\n"
        "1. NEVER invent graph edges or callers outside the provided VERIFIED_EDGES list.\n"
        "2. Explicitly distinguish verified static relationships from inferred runtime risks.\n"
        "3. Return ONLY valid raw JSON (no markdown fences) matching this exact schema:\n"
        "{\n"
        '  "verified_facts": [{"caller_id": "...", "callee_id": "...", "evidence_location": "file:line", "summary": "..."}],\n'
        '  "inferred_risks": [{"affected_symbol": "...", "hypothesis": "...", "confidence_note": "Inferred from source excerpt"}],\n'
        '  "suggested_test_checks": ["..."]\n'
        "}\n\n"
        f"TARGET SYMBOL ({payload.target_symbol.symbol_id}):\n"
        f"{payload.target_symbol.source_excerpt}\n\n"
        f"VERIFIED_EDGES:\n{json.dumps(payload.verified_edges, indent=2)}\n\n"
        f"CALLER_EXCERPTS:\n{json.dumps(payload.caller_excerpts, indent=2)}\n"
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from model output, stripping markdown fences if present."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"`{3}(?:json)?\s*(\{.*?\})\s*`{3}", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected top-level JSON object from Gemma response.")
    return parsed


def validate_and_ground_gemma_output(
    raw_output: Dict[str, Any],
    payload: GemmaPromptPayload,
    provider_name: str,
    model_name: str,
) -> ValidatedGemmaExplanation:
    """
    Validate raw model JSON against the deterministic graph.
    Strips any hallucinated caller->callee edge that does not exist in `payload.verified_edges`.
    """
    warnings: List[str] = []
    allowed_pairs: Set[Tuple[str, str]] = {
        (e["caller_id"], e["callee_id"]) for e in payload.verified_edges
    }
    allowed_symbols: Set[str] = set(payload.transitive_callers) | {
        payload.target_symbol.symbol_id
    }

    verified_facts: List[VerifiedFact] = []
    for item in raw_output.get("verified_facts", []):
        if not isinstance(item, dict):
            continue
        c_id = str(item.get("caller_id", ""))
        t_id = str(item.get("callee_id", ""))
        if (c_id, t_id) in allowed_pairs:
            verified_facts.append(
                VerifiedFact(
                    caller_id=c_id,
                    callee_id=t_id,
                    evidence_location=str(item.get("evidence_location", "")),
                    summary=str(item.get("summary", "")),
                )
            )
        else:
            warnings.append(
                f"Rejected unverified model-claimed edge: {c_id} -> {t_id}"
            )

    inferred_risks: List[InferredRisk] = []
    for risk in raw_output.get("inferred_risks", []):
        if not isinstance(risk, dict):
            continue
        sym = str(risk.get("affected_symbol", ""))
        if sym and sym not in allowed_symbols:
            warnings.append(
                f"Flagged inferred risk referencing symbol outside reachability set: {sym}"
            )
            continue
        inferred_risks.append(
            InferredRisk(
                affected_symbol=sym or payload.target_symbol.symbol_id,
                hypothesis=str(risk.get("hypothesis", "")),
                confidence_note=str(
                    risk.get(
                        "confidence_note",
                        "Model inference (non-deterministic; verify manually)",
                    )
                ),
            )
        )

    suggestions = [
        str(s) for s in raw_output.get("suggested_test_checks", []) if s
    ]

    disclaimer = (
        f"Static AST analysis is partial: {len(payload.verified_edges)} verified call edge(s) "
        f"in subgraph, and {payload.unresolved_count} unresolved dynamic call(s) across repository."
    )

    return ValidatedGemmaExplanation(
        provider_name=provider_name,
        model_name=model_name,
        target_symbol_id=payload.target_symbol.symbol_id,
        verified_facts=verified_facts,
        inferred_risks=inferred_risks,
        suggested_test_checks=suggestions,
        coverage_disclaimer=disclaimer,
        validation_warnings=warnings,
    )


class ConfigurableGemmaProvider:
    """
    Calls an eligible Gemma 4 inference API via OpenRouter (or any OpenAI-compatible
    endpoint / Google Generative Language API) using environment variables or defaults:
      - GEMMA_API_BASE (default: 'https://openrouter.ai/api/v1')
      - GEMMA_MODEL    (default: 'google/gemma-4-26b-a4b-it:free')
      - GEMMA_API_KEY  (default: configured OpenRouter key or environment override)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_sec: int = 30,
    ) -> None:
        self.api_key = api_key if api_key is not None else get_configured_api_key()
        self.model_name = model_name or os.environ.get(
            "GEMMA_MODEL", DEFAULT_GEMMA_MODEL
        )
        self.api_base = (
            api_base
            if api_base is not None
            else os.environ.get("GEMMA_API_BASE", DEFAULT_GEMMA_API_BASE)
        )
        self.timeout_sec = timeout_sec

    def generate_explanation(self, payload: GemmaPromptPayload) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("No GEMMA_API_KEY configured.")

        prompt_text = format_gemma_system_prompt(payload)

        if self.api_base and "generativelanguage.googleapis.com" not in self.api_base:
            # OpenRouter / OpenAI-compatible /v1/chat/completions endpoint
            url = f"{self.api_base.rstrip('/')}/chat/completions"
            body = json.dumps(
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.1,
                }
            ).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ripple-change-impact",
                "X-Title": "Ripple Change-Impact Explorer",
            }
        else:
            # Native Google Generative Language API endpoint
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model_name}:generateContent?key={self.api_key}"
            )
            body = json.dumps(
                {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                    },
                }
            ).encode("utf-8")
            headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8"))

        if "choices" in raw_resp and raw_resp["choices"]:
            text = raw_resp["choices"][0]["message"]["content"]
        elif "candidates" in raw_resp and raw_resp["candidates"]:
            text = raw_resp["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise ValueError(f"Unrecognized response structure from Gemma endpoint: {raw_resp}")

        return _extract_json_object(text)


class EvidenceGroundedFallbackProvider:
    """
    Deterministic evidence synthesizer used when offline mode is selected
    or during unit testing. Never invents edges.
    """

    model_name: str = "gemma-4-offline-evidence-synthesizer"

    def generate_explanation(self, payload: GemmaPromptPayload) -> Dict[str, Any]:
        facts = [
            {
                "caller_id": e["caller_id"],
                "callee_id": e["callee_id"],
                "evidence_location": f"{e['file_path']}:{e['line']}",
                "summary": f"`{e['caller_id']}` statically calls `{e['callee_id']}` via `{e['snippet']}`.",
            }
            for e in payload.verified_edges
        ]

        risks = []
        for caller_id in payload.direct_callers:
            risks.append(
                {
                    "affected_symbol": caller_id,
                    "hypothesis": (
                        f"Changes to `{payload.target_symbol.name}` return type, raised exceptions, "
                        f"or parameter defaults may directly alter `{caller_id}`."
                    ),
                    "confidence_note": "Inferred from direct static call site (verify signature compatibility).",
                }
            )
        for trans_id in payload.transitive_callers:
            if trans_id not in payload.direct_callers:
                risks.append(
                    {
                        "affected_symbol": trans_id,
                        "hypothesis": (
                            f"Transitive caller `{trans_id}` depends on intermediate callers of "
                            f"`{payload.target_symbol.name}` and may observe downstream behavior shifts."
                        ),
                        "confidence_note": "Inferred from transitive call chain.",
                    }
                )

        test_checks = [
            f"Run `{test_id}` to verify behavior along its call path to `{payload.target_symbol.symbol_id}`."
            for test_id in payload.candidate_tests
        ]
        if not test_checks:
            test_checks.append(
                f"No existing `test_*` caller reached `{payload.target_symbol.symbol_id}`; "
                "consider adding a direct unit test."
            )

        return {
            "verified_facts": facts,
            "inferred_risks": risks,
            "suggested_test_checks": test_checks,
        }


def generate_grounded_explanation(
    bundle: DependencyGraphBundle,
    impact: ImpactReport,
    provider: Optional[InferenceProvider] = None,
) -> ValidatedGemmaExplanation:
    """Build bounded prompt, query Gemma 4 (or fallback gently on error), and validate."""
    payload = build_gemma_prompt_payload(bundle, impact)

    active_provider = provider
    provider_label = "CustomGemmaProvider"
    model_label = getattr(provider, "model_name", DEFAULT_GEMMA_MODEL)

    if active_provider is None:
        active_key = get_configured_api_key()
        if active_key:
            cfg = ConfigurableGemmaProvider(api_key=active_key)
            active_provider = cfg
            provider_label = f"ConfigurableGemmaProvider ({cfg.api_base})"
            model_label = cfg.model_name
        else:
            fb = EvidenceGroundedFallbackProvider()
            active_provider = fb
            provider_label = "EvidenceGroundedFallbackProvider (Offline Mode - No API Key Set)"
            model_label = fb.model_name

    try:
        raw_json = active_provider.generate_explanation(payload)
    except Exception as exc:
        fb = EvidenceGroundedFallbackProvider()
        raw_json = fb.generate_explanation(payload)
        explained = validate_and_ground_gemma_output(
            raw_json,
            payload,
            provider_name=f"{provider_label} -> Fallback ({type(exc).__name__})",
            model_name=fb.model_name,
        )
        explained.validation_warnings.append(
            f"Live Gemma API call failed ({exc}); fell back to deterministic offline synthesizer."
        )
        return explained

    return validate_and_ground_gemma_output(
        raw_json, payload, provider_name=provider_label, model_name=model_label
    )


# ============================================================================
# MILESTONE 5: TINY SAMPLE REPOSITORY & PYTEST UNIT TESTS
# ============================================================================

SAMPLE_PROJECT_FILES: Dict[str, str] = {
    "billing/__init__.py": '"""Sample billing package."""\n',
    "billing/tax.py": (
        "def get_tax_rate(region: str) -> float:\n"
        '    """Base leaf function that returns a regional tax multiplier."""\n'
        "    rates = {'US': 0.08, 'EU': 0.20, 'IN': 0.18}\n"
        "    return rates.get(region, 0.0)\n"
        "\n"
        "\n"
        "def compute_tax(subtotal: float, region: str) -> float:\n"
        '    """Direct caller of get_tax_rate."""\n'
        "    rate = get_tax_rate(region)\n"
        "    return round(subtotal * rate, 2)\n"
    ),
    "billing/invoice.py": (
        "from billing.tax import compute_tax\n"
        "\n"
        "\n"
        "class InvoiceProcessor:\n"
        "    def format_line(self, label: str, amount: float) -> str:\n"
        "        return f'{label}: ${amount:.2f}'\n"
        "\n"
        "    def build_invoice(self, subtotal: float, region: str) -> dict:\n"
        "        tax = compute_tax(subtotal, region)\n"
        "        total = subtotal + tax\n"
        "        summary = self.format_line('Total', total)\n"
        "        return {'subtotal': subtotal, 'tax': tax, 'total': total, 'summary': summary}\n"
    ),
    "api/checkout.py": (
        "from billing.invoice import InvoiceProcessor\n"
        "from billing import tax\n"
        "\n"
        "\n"
        "def handle_checkout(cart_subtotal: float, region: str, plugin=None) -> dict:\n"
        "    processor = InvoiceProcessor()\n"
        "    invoice = processor.build_invoice(cart_subtotal, region)\n"
        "    if plugin is not None:\n"
        "        # Intentional dynamic call to verify unresolved call reporting\n"
        "        hook = getattr(plugin, 'on_checkout')\n"
        "        hook(invoice)\n"
        "    return invoice\n"
        "\n"
        "\n"
        "def preview_regional_tax(region: str) -> float:\n"
        "    return tax.get_tax_rate(region)\n"
    ),
    "tests/test_checkout.py": (
        "from api.checkout import handle_checkout, preview_regional_tax\n"
        "\n"
        "\n"
        "def test_end_to_end_checkout():\n"
        "    result = handle_checkout(100.0, 'US')\n"
        "    assert result['total'] == 108.0\n"
        "\n"
        "\n"
        "def test_tax_preview():\n"
        "    assert preview_regional_tax('EU') == 0.20\n"
    ),
}


def materialize_sample_repo(target_dir: Path) -> Path:
    """Write the tiny sample repository to `target_dir` for analysis or testing."""
    for rel_path, content in SAMPLE_PROJECT_FILES.items():
        full_path = target_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    return target_dir


def analyze_sample_project() -> DependencyGraphBundle:
    """Ingest, parse, and build the dependency graph for the built-in sample project."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = materialize_sample_repo(Path(tmp))
        ingestion = ingest_repository(repo_dir)
        parsed = parse_ingested_repository(ingestion)
        return build_dependency_graph(parsed)


def test_milestone1_workspace_inspection() -> None:
    """Verify workspace inspection reports Python version and core packages."""
    report = inspect_workspace()
    assert report.python_ok is True
    assert report.dependencies["ast"] is True
    assert report.dependencies["networkx"] is True


def test_milestone2_ast_parser_extracts_symbols_and_unresolved_calls() -> None:
    """Verify AST parser discovers functions, methods, classes, imports, and unresolved calls."""
    bundle = analyze_sample_project()
    symbols = bundle.parsed.symbols

    expected_symbols = {
        "billing.tax.get_tax_rate",
        "billing.tax.compute_tax",
        "billing.invoice.InvoiceProcessor",
        "billing.invoice.InvoiceProcessor.format_line",
        "billing.invoice.InvoiceProcessor.build_invoice",
        "api.checkout.handle_checkout",
        "api.checkout.preview_regional_tax",
        "tests.test_checkout.test_end_to_end_checkout",
        "tests.test_checkout.test_tax_preview",
    }
    assert expected_symbols.issubset(set(symbols.keys()))

    tax_fn = symbols["billing.tax.get_tax_rate"]
    assert tax_fn.start_line == 1
    assert tax_fn.end_line == 4
    assert tax_fn.file_path == "billing/tax.py"

    unresolved_reasons = [
        (u.caller_id, u.expr_description) for u in bundle.unresolved_calls
    ]
    assert any(
        caller == "api.checkout.handle_checkout" and "getattr" in expr
        for caller, expr in unresolved_reasons
    )


def test_milestone3_directed_graph_edges_and_relationships() -> None:
    """Verify directed CALLS edges point from caller -> callee with explicit metadata."""
    bundle = analyze_sample_project()
    g = bundle.graph

    assert g.has_edge("billing.tax.compute_tax", "billing.tax.get_tax_rate")
    assert (
        g.edges["billing.tax.compute_tax", "billing.tax.get_tax_rate"]["relationship"]
        == "CALLS"
    )

    assert g.has_edge(
        "billing.invoice.InvoiceProcessor.build_invoice",
        "billing.tax.compute_tax",
    )

    assert g.has_edge(
        "billing.invoice.InvoiceProcessor.build_invoice",
        "billing.invoice.InvoiceProcessor.format_line",
    )


def test_milestone4_reverse_reachability_excludes_changed_node() -> None:
    """
    Verify reverse reachability returns all direct & transitive callers, candidate tests,
    and impact paths while strictly excluding the changed target node from affected_nodes.
    """
    bundle = analyze_sample_project()
    target = "billing.tax.get_tax_rate"
    impact = compute_reverse_reachability(bundle, target)

    assert target not in impact.affected_nodes
    assert target not in impact.direct_callers

    assert impact.direct_callers == [
        "api.checkout.preview_regional_tax",
        "billing.tax.compute_tax",
    ]

    expected_affected = [
        "api.checkout.handle_checkout",
        "api.checkout.preview_regional_tax",
        "billing.invoice.InvoiceProcessor.build_invoice",
        "billing.tax.compute_tax",
        "tests.test_checkout.test_end_to_end_checkout",
        "tests.test_checkout.test_tax_preview",
    ]
    assert impact.affected_nodes == expected_affected

    assert impact.candidate_tests == [
        "tests.test_checkout.test_end_to_end_checkout",
        "tests.test_checkout.test_tax_preview",
    ]

    expected_chain = [
        "tests.test_checkout.test_end_to_end_checkout",
        "api.checkout.handle_checkout",
        "billing.invoice.InvoiceProcessor.build_invoice",
        "billing.tax.compute_tax",
        "billing.tax.get_tax_rate",
    ]
    assert expected_chain in impact.impact_paths


def test_milestone7_gemma_validator_and_offline_fallback() -> None:
    """Verify the Gemma validator strips fabricated edges and offline fallback works without an API key."""
    bundle = analyze_sample_project()
    impact = compute_reverse_reachability(bundle, "billing.tax.get_tax_rate")
    payload = build_gemma_prompt_payload(bundle, impact)

    # 1. Verify offline fallback works without an API key
    offline_provider = EvidenceGroundedFallbackProvider()
    offline_explained = generate_grounded_explanation(
        bundle, impact, provider=offline_provider
    )
    assert len(offline_explained.verified_facts) == len(impact.supporting_edges)
    assert len(offline_explained.inferred_risks) > 0
    assert len(offline_explained.validation_warnings) == 0

    # 2. Verify validator rejects hallucinated edges
    fake_model_response = {
        "verified_facts": [
            {
                "caller_id": "billing.tax.compute_tax",
                "callee_id": "billing.tax.get_tax_rate",
                "evidence_location": "billing/tax.py:9",
                "summary": "Valid edge",
            },
            {
                "caller_id": "invented.module.ghost_caller",
                "callee_id": "billing.tax.get_tax_rate",
                "evidence_location": "ghost.py:1",
                "summary": "Hallucinated edge that must be rejected",
            },
        ],
        "inferred_risks": [
            {
                "affected_symbol": "billing.tax.compute_tax",
                "hypothesis": "Precision change could alter rounding.",
                "confidence_note": "Inferred",
            },
            {
                "affected_symbol": "unrelated.symbol",
                "hypothesis": "Should be filtered out",
                "confidence_note": "Inferred",
            },
        ],
        "suggested_test_checks": ["Run pytest on tests/test_checkout.py"],
    }

    validated = validate_and_ground_gemma_output(
        fake_model_response,
        payload,
        provider_name="MockGemmaProvider",
        model_name="google/gemma-4-26b-a4b-it:free",
    )

    assert len(validated.verified_facts) == 1
    assert validated.verified_facts[0].caller_id == "billing.tax.compute_tax"
    assert len(validated.inferred_risks) == 1
    assert len(validated.validation_warnings) == 2


def test_milestone8_safe_ingestion_enforces_limits() -> None:
    """Verify repository ingestion skips oversized files and handles syntax errors safely."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (root / "huge.py").write_text("x = 1\n" * 5000, encoding="utf-8")
        (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")

        limits = RepoLimits(max_files=10, max_file_bytes=500)
        ingestion = ingest_repository(root, limits=limits)

        ingested_names = [f.rel_path for f in ingestion.files]
        assert "good.py" in ingested_names
        assert "huge.py" not in ingested_names
        assert any(name == "huge.py" for name, _ in ingestion.skipped_files)

        parsed = parse_ingested_repository(ingestion)
        assert "good.ok" in parsed.symbols
        assert len(parsed.syntax_errors) == 1
        assert parsed.syntax_errors[0][0] == "broken.py"


# ============================================================================
# MILESTONE 6: MINIMAL STREAMLIT UI
# ============================================================================

def run_streamlit_app() -> None:
    """Render the minimal functional Streamlit interface for Ripple."""
    import streamlit as st

    st.set_page_config(page_title="Ripple | Change-Impact Explorer", layout="wide")
    st.title("Ripple: Static Python Change-Impact Explorer")
    st.caption(
        "Deterministic `ast` + `NetworkX` reverse-reachability analyzer with grounded Gemma 4 explanations. "
        "Analyzed repositories are never executed."
    )

    with st.sidebar:
        st.header("1. Repository Source")
        source_mode = st.radio(
            "Select analysis target:",
            options=["Built-in Sample Project", "Local Directory Path"],
            index=0,
        )
        local_path_str = ""
        if source_mode == "Local Directory Path":
            local_path_str = st.text_input(
                "Local Python repository path:", value=str(Path.cwd())
            )

        st.header("2. Gemma 4 Configuration")
        use_offline = st.checkbox("Force Offline Fallback Mode", value=False)
        api_key_input = st.text_input(
            "OpenRouter API Key (`GEMMA_API_KEY`):",
            value=get_configured_api_key(),
            type="password",
        )
        api_base_input = st.text_input(
            "API Base (`GEMMA_API_BASE`):",
            value=os.environ.get("GEMMA_API_BASE", DEFAULT_GEMMA_API_BASE),
        )
        model_choice = st.text_input(
            "Gemma Model (`GEMMA_MODEL`):",
            value=os.environ.get("GEMMA_MODEL", DEFAULT_GEMMA_MODEL),
        )
        os.environ["GEMMA_API_BASE"] = api_base_input.strip()
        os.environ["GEMMA_MODEL"] = model_choice.strip()

    try:
        if source_mode == "Built-in Sample Project":
            bundle = analyze_sample_project()
            ingestion_note = "Loaded built-in 5-file sample repository (`billing`, `api`, `tests`)."
        else:
            target_path = Path(local_path_str).expanduser()
            ingestion = ingest_repository(target_path)
            parsed = parse_ingested_repository(ingestion)
            bundle = build_dependency_graph(parsed)
            ingestion_note = (
                f"Ingested {len(ingestion.files)} Python file(s) from `{ingestion.root_path}` "
                f"({len(ingestion.skipped_files)} skipped, {len(parsed.syntax_errors)} syntax error(s))."
            )
    except Exception as exc:
        st.error(f"Repository ingestion failed: {exc}")
        return

    st.info(ingestion_note)

    callable_symbols = sorted(
        sym_id
        for sym_id, sym in bundle.parsed.symbols.items()
        if sym.kind in ("function", "method")
    )
    if not callable_symbols:
        st.warning("No function or method definitions discovered in repository.")
        return

    default_idx = (
        callable_symbols.index("billing.tax.get_tax_rate")
        if "billing.tax.get_tax_rate" in callable_symbols
        else 0
    )
    selected_symbol = st.selectbox(
        "Select a changed function or method to inspect reverse reachability:",
        options=callable_symbols,
        index=default_idx,
    )

    impact = compute_reverse_reachability(bundle, selected_symbol)
    target_meta = bundle.parsed.symbols[selected_symbol]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Direct Callers", len(impact.direct_callers))
    col2.metric("Transitive Affected Nodes", len(impact.affected_nodes))
    col3.metric("Candidate Tests", len(impact.candidate_tests))
    col4.metric("Unresolved Calls (Repo)", len(bundle.unresolved_calls))

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Target Symbol Source Evidence")
        st.markdown(
            f"**Symbol:** `{target_meta.symbol_id}`  \n"
            f"**Location:** `{target_meta.file_path}:{target_meta.start_line}-{target_meta.end_line}`"
        )
        st.code(target_meta.source_excerpt, language="python")

        st.subheader("Direct Callers (1-Hop)")
        if impact.direct_callers:
            for c_id in impact.direct_callers:
                st.markdown(f"- `{c_id}`")
        else:
            st.write("No statically discovered direct callers.")

        st.subheader("Candidate Tests")
        if impact.candidate_tests:
            for t_id in impact.candidate_tests:
                st.markdown(f"- `{t_id}`")
        else:
            st.write("No `test_*` functions reached this symbol statically.")

    with right_col:
        st.subheader("Transitive Impact Paths (Caller -> ... -> Target)")
        if impact.impact_paths:
            for path in impact.impact_paths:
                st.code(" -> ".join(path), language="text")
        else:
            st.write("No incoming call paths found.")

        st.subheader("Verified Supporting Graph Edges")
        if impact.supporting_edges:
            edge_rows = [asdict(e) for e in impact.supporting_edges]
            st.dataframe(edge_rows, use_container_width=True)
        else:
            st.write("No supporting call edges for this symbol.")

    st.divider()
    st.subheader("Gemma 4 Grounded Change-Impact Explanation")

    active_provider: Optional[InferenceProvider] = None
    if use_offline or not api_key_input.strip():
        active_provider = EvidenceGroundedFallbackProvider()
    else:
        active_provider = ConfigurableGemmaProvider(
            api_key=api_key_input.strip(),
            model_name=model_choice.strip(),
            api_base=api_base_input.strip(),
        )

    explanation = generate_grounded_explanation(bundle, impact, provider=active_provider)
    st.caption(
        f"Provider: `{explanation.provider_name}` | Model: `{explanation.model_name}` | "
        f"{explanation.coverage_disclaimer}"
    )

    if explanation.validation_warnings:
        for warn in explanation.validation_warnings:
            st.warning(warn)

    f_col, r_col = st.columns(2)
    with f_col:
        st.markdown("#### Deterministic Verified Facts (Graph-Grounded)")
        for fact in explanation.verified_facts:
            st.markdown(
                f"- **`{fact.caller_id}` -> `{fact.callee_id}`** (`{fact.evidence_location}`): {fact.summary}"
            )
    with r_col:
        st.markdown("#### Inferred Risks & Suggestions (Non-Deterministic)")
        for risk in explanation.inferred_risks:
            st.markdown(
                f"- **`{risk.affected_symbol}`**: {risk.hypothesis} *({risk.confidence_note})*"
            )
        st.markdown("**Suggested Test Verification:**")
        for check in explanation.suggested_test_checks:
            st.markdown(f"- {check}")

    with st.expander("View Repository Unresolved / Dynamic Calls (Static Coverage Limits)"):
        if bundle.unresolved_calls:
            st.dataframe(
                [asdict(u) for u in bundle.unresolved_calls],
                use_container_width=True,
            )
        else:
            st.write("No unresolved calls recorded.")


# ============================================================================
# CLI & SELF-TEST RUNNER
# ============================================================================

def _is_running_in_streamlit() -> bool:
    """Return True when executed inside an active `streamlit run` session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _run_cli() -> int:
    if _is_running_in_streamlit():
        run_streamlit_app()
        return 0

    parser = argparse.ArgumentParser(
        description="Ripple: Static Python Change-Impact Explorer CLI"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Milestone 1 workspace and dependency verification.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run Milestones 2-7 on the built-in sample project and print the JSON report.",
    )
    parser.add_argument(
        "--materialize-sample",
        type=str,
        default="",
        help="Write the sample repository to the specified directory path.",
    )
    args, _ = parser.parse_known_args()

    if args.check_env:
        rep = inspect_workspace()
        print(json.dumps(asdict(rep), indent=2))
        return 0

    if args.materialize_sample:
        out = materialize_sample_repo(Path(args.materialize_sample))
        print(f"Materialized sample repository to: {out.resolve()}")
        return 0

    if args.demo:
        bundle = analyze_sample_project()
        impact = compute_reverse_reachability(bundle, "billing.tax.get_tax_rate")
        explanation = generate_grounded_explanation(bundle, impact)
        output = {
            "environment": asdict(inspect_workspace()),
            "target_symbol": impact.target_symbol_id,
            "direct_callers": impact.direct_callers,
            "affected_nodes": impact.affected_nodes,
            "candidate_tests": impact.candidate_tests,
            "impact_paths": impact.impact_paths,
            "unresolved_calls": [asdict(u) for u in bundle.unresolved_calls],
            "gemma_explanation": asdict(explanation),
        }
        print(json.dumps(output, indent=2))
        return 0

    print("=== Milestone 1: Environment & Dependency Check ===")
    env_rep = inspect_workspace()
    print(json.dumps(asdict(env_rep), indent=2))

    print("\n=== Running Built-in Verification Suite (Milestones 1-8) ===")
    test_funcs = [
        test_milestone1_workspace_inspection,
        test_milestone2_ast_parser_extracts_symbols_and_unresolved_calls,
        test_milestone3_directed_graph_edges_and_relationships,
        test_milestone4_reverse_reachability_excludes_changed_node,
        test_milestone7_gemma_validator_and_offline_fallback,
        test_milestone8_safe_ingestion_enforces_limits,
    ]
    for fn in test_funcs:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print("\nAll 6 verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
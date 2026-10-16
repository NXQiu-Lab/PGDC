"""Generate leaf README files and aggregated package docs."""

from __future__ import annotations

import ast
from datetime import date
from dataclasses import dataclass
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "atom_dataset_index"
DOCS_ROOT = REPO_ROOT / "docs"

SKIP_DIRS = {"__pycache__"}
TOP_LEVEL_MODULES = {
    "config",
    "io",
    "domain",
    "metric_core",
    "metrics",
    "profiling",
    "benchmark",
    "presentation",
    "utils",
}

HIGH_LEVEL_DOCS: tuple[str, ...] = (
    "docs/project_design_and_paper_plan.md",
    "docs/technical_architecture_plan.md",
    "docs/module_specification.md",
    "docs/python_package_and_class_spec.md",
)


@dataclass(slots=True)
class SymbolInfo:
    name: str
    kind: str
    signature: str
    docstring: str


@dataclass(slots=True)
class LeafDoc:
    module_id: str
    module_version: str
    project_version: str
    last_updated: str
    generation_mode: str
    source_files: list[str]
    upstream_modules: list[str]
    downstream_modules: list[str]
    summary: str
    symbols: list[SymbolInfo]
    readme_path: Path


def _iter_leaf_dirs() -> list[Path]:
    leaf_dirs: list[Path] = []
    for directory in SRC_ROOT.rglob("*"):
        if not directory.is_dir() or directory.name in SKIP_DIRS:
            continue
        py_files = [path for path in directory.glob("*.py") if path.name != "__init__.py"]
        child_dirs = [
            path for path in directory.iterdir()
            if path.is_dir() and path.name not in SKIP_DIRS
        ]
        if py_files and not child_dirs:
            leaf_dirs.append(directory)
    return sorted(leaf_dirs)


def _signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    args = [argument.arg for argument in node.args.args]
    return f"{node.name}({', '.join(args)})"


def _extract_symbols(py_file: Path) -> list[SymbolInfo]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    symbols: list[SymbolInfo] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            symbols.append(
                SymbolInfo(
                    name=node.name,
                    kind="class",
                    signature=_signature(node),
                    docstring=ast.get_docstring(node) or "",
                )
            )
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            symbols.append(
                SymbolInfo(
                    name=node.name,
                    kind="function",
                    signature=_signature(node),
                    docstring=ast.get_docstring(node) or "",
                )
            )
    return symbols


def _summarize_leaf(directory: Path) -> str:
    relative = directory.relative_to(SRC_ROOT)
    parts = relative.parts
    if not parts:
        return "Root package helpers."
    if parts[0] == "metric_core":
        return "Shared metric infrastructure reused by algorithm-level metrics."
    if parts[0] == "metrics":
        return "Algorithm-level metric implementations."
    if parts[0] == "presentation":
        return "Presentation outputs including exports and manifests."
    return f"Package leaf module for {'/'.join(parts)}."


def _upstream_for(parts: tuple[str, ...]) -> list[str]:
    if not parts:
        return []
    if parts[0] == "metrics":
        return ["atom_dataset_index.metric_core", "atom_dataset_index.domain"]
    if parts[0] == "profiling":
        return ["atom_dataset_index.metrics", "atom_dataset_index.metric_core"]
    if parts[0] == "benchmark":
        return ["atom_dataset_index.profiling", "atom_dataset_index.domain"]
    if parts[0] == "presentation":
        return ["atom_dataset_index.benchmark", "atom_dataset_index.profiling"]
    return ["atom_dataset_index.domain"] if parts[0] in {"io", "metric_core"} else []


def _downstream_for(parts: tuple[str, ...]) -> list[str]:
    if not parts:
        return []
    if parts[0] == "metric_core":
        return ["atom_dataset_index.metrics", "atom_dataset_index.profiling"]
    if parts[0] == "metrics":
        return ["atom_dataset_index.profiling", "atom_dataset_index.benchmark"]
    if parts[0] == "profiling":
        return ["atom_dataset_index.benchmark", "atom_dataset_index.presentation"]
    if parts[0] == "benchmark":
        return ["atom_dataset_index.presentation"]
    return []


def _build_leaf_doc(directory: Path, project_version: str, today: str) -> LeafDoc:
    relative = directory.relative_to(SRC_ROOT)
    py_files = sorted(path for path in directory.glob("*.py") if path.name != "__init__.py")
    symbols: list[SymbolInfo] = []
    for py_file in py_files:
        symbols.extend(_extract_symbols(py_file))
    module_id = "atom_dataset_index." + ".".join(relative.parts)
    return LeafDoc(
        module_id=module_id,
        module_version=f"{project_version}-r1",
        project_version=project_version,
        last_updated=today,
        generation_mode="auto",
        source_files=[str(path.relative_to(REPO_ROOT)) for path in py_files],
        upstream_modules=_upstream_for(relative.parts),
        downstream_modules=_downstream_for(relative.parts),
        summary=_summarize_leaf(directory),
        symbols=symbols,
        readme_path=directory / "README.md",
    )


def _render_leaf_readme(leaf: LeafDoc) -> str:
    lines = [
        f"# {leaf.module_id}",
        "",
        "## Metadata",
        "",
        f"- `module_id`: `{leaf.module_id}`",
        f"- `module_version`: `{leaf.module_version}`",
        f"- `project_version`: `{leaf.project_version}`",
        f"- `last_updated`: `{leaf.last_updated}`",
        f"- `generation_mode`: `{leaf.generation_mode}`",
        f"- `source_files`: {', '.join(f'`{path}`' for path in leaf.source_files)}",
        f"- `upstream_modules`: {', '.join(f'`{path}`' for path in leaf.upstream_modules) or '`none`'}",
        f"- `downstream_modules`: {', '.join(f'`{path}`' for path in leaf.downstream_modules) or '`none`'}",
        "",
        "## Module Purpose",
        "",
        leaf.summary,
        "",
        "## Public Symbols",
        "",
    ]
    if not leaf.symbols:
        lines.append("- No public classes or functions were discovered.")
    else:
        for symbol in leaf.symbols:
            description = symbol.docstring or "No docstring provided."
            lines.append(f"- `{symbol.signature}`")
            lines.append(f"  {description}")
    lines.extend(
        [
            "",
            "## I/O Contract",
            "",
            "- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.",
            "- Outputs: Stable objects or helper values returned by the public symbols in this directory.",
            "- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.",
            "",
            "## Update Rule",
            "",
            "- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_leaf_metadata(readme_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- `") and "`: `" in line:
            left, right = line.split("`: `", 1)
            key = left.removeprefix("- `")
            value = right.removesuffix("`")
            metadata[key] = value
    return metadata


def _read_project_version(default: str = "0.5.0") -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return default
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version.strip():
        return version
    return default


def _render_package_doc(leaf_docs: list[LeafDoc], project_version: str, today: str) -> str:
    lines = [
        "# Python 子目录与类说明文档",
        "",
        "## Metadata",
        "",
        "- `document_id`: `docs/python_package_and_class_spec.md`",
        f"- `document_version`: `{project_version}-r1`",
        f"- `project_version`: `{project_version}`",
        f"- `last_updated`: `{today}`",
        "- `generation_mode`: `hybrid`",
        "- `derived_from`: `src/**/README.md`",
        "",
        "## 包结构总览",
        "",
        "- `config/`: configuration models and defaults",
        "- `io/`: ASE DB readers and structured writers",
        "- `domain/`: stable shared domain models",
        "- `metric_core/`: metric infrastructure, validation, registry, descriptors",
        "- `metrics/`: algorithm-level metric implementations",
        "- `profiling/`: profile orchestration",
        "- `benchmark/`: SchNetPack baseline orchestration",
        "- `presentation/`: manifests, exports, and presentation outputs",
        "- `tasks.py`: runtime task registry and dynamic output directory creation",
        "- `utils/`: repository utilities including doc generation",
        "",
        "## 叶子目录索引",
        "",
    ]
    for leaf in leaf_docs:
        relative = leaf.readme_path.relative_to(REPO_ROOT)
        lines.append(f"- `{leaf.module_id}` -> `{relative}`")
    lines.extend(["", "## 叶子文档汇总", ""])
    for leaf in leaf_docs:
        lines.append(f"### `{leaf.module_id}`")
        lines.append("")
        lines.append(f"- Version: `{leaf.module_version}`")
        lines.append(f"- Sources: {', '.join(f'`{path}`' for path in leaf.source_files)}")
        if leaf.symbols:
            lines.append(
                "- Symbols: "
                + ", ".join(f"`{symbol.name}`" for symbol in leaf.symbols)
            )
        else:
            lines.append("- Symbols: `none`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _sync_version_manifest(project_version: str, today: str) -> None:
    revisions = "\n".join(
        f"  - id: {doc_id}\n    revision: r1"
        for doc_id in HIGH_LEVEL_DOCS
    )
    content = (
        f"project_version: {project_version}\n"
        f"last_updated: {today}\n"
        "documents:\n"
        f"{revisions}\n"
    )
    (DOCS_ROOT / "version_manifest.yaml").write_text(content, encoding="utf-8")


def main(project_version: str | None = None, today: str | None = None) -> int:
    resolved_project_version = project_version or _read_project_version()
    resolved_today = today or date.today().isoformat()
    leaf_docs = [
        _build_leaf_doc(directory, resolved_project_version, resolved_today)
        for directory in _iter_leaf_dirs()
    ]
    for leaf in leaf_docs:
        leaf.readme_path.write_text(_render_leaf_readme(leaf), encoding="utf-8")

    package_doc = DOCS_ROOT / "python_package_and_class_spec.md"
    package_doc.write_text(
        _render_package_doc(leaf_docs, resolved_project_version, resolved_today),
        encoding="utf-8",
    )
    _sync_version_manifest(resolved_project_version, resolved_today)
    return 0

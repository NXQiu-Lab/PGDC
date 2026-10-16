"""Core package for atom-dataset-index."""

__version__ = "0.5.0"

from .tasks import TaskDefinition, build_task_name, list_task_families


def build_parser():
    """Lazily expose CLI parser builder without importing cli at package import time."""

    from .cli import build_parser as _build_parser

    return _build_parser()


def main(argv=None):
    """Lazily expose CLI main entrypoint."""

    from .cli import main as _main

    return _main(argv)


__all__ = [
    "TaskDefinition",
    "__version__",
    "build_parser",
    "build_task_name",
    "list_task_families",
    "main",
]

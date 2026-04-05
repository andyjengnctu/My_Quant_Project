from __future__ import annotations


def main(*args, **kwargs):
    from .main import main as _main

    return _main(*args, **kwargs)


def build_workbench_spec():
    from .workbench import build_workbench_spec as _build_workbench_spec

    return _build_workbench_spec()


__all__ = ["build_workbench_spec", "main"]

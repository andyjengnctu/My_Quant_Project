from __future__ import annotations


def main(*args, **kwargs):
    from .main import main as _main

    return _main(*args, **kwargs)


def __getattr__(name):
    if name in {"OptimizerSession", "close_study_storage"}:
        from . import session as session_module

        value = getattr(session_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OptimizerSession",
    "close_study_storage",
    "main",
]

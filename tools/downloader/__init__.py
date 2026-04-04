from __future__ import annotations


def main(*args, **kwargs):
    from .main import main as _main

    return _main(*args, **kwargs)


def smart_download_vip_data(*args, **kwargs):
    from .main import smart_download_vip_data as _smart_download_vip_data

    return _smart_download_vip_data(*args, **kwargs)


def __getattr__(name):
    if name in {"SAVE_DIR", "FINMIND_PRICE_DATASET", "dl", "time"}:
        from . import runtime as rt

        value = getattr(rt, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SAVE_DIR",
    "FINMIND_PRICE_DATASET",
    "dl",
    "time",
    "smart_download_vip_data",
    "main",
]

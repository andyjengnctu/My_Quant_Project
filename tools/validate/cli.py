import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main(argv=None, environ=None):
    from tools.validate import main as validate_main

    return validate_main(argv=argv, environ=environ)


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())

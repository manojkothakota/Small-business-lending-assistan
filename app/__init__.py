"""Loads a local .env file (KEY=value per line) before anything else reads the environment."""
import os
from pathlib import Path

_f = Path(".env")
if _f.exists():
    for _l in _f.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

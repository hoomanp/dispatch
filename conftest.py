"""Pytest bootstrap for the repository root.

Two import styles are used across the suite and both must work no matter how
pytest is invoked:

  * ``from rate_limiter import ...`` — each test file inserts ``dispatcher/``
    onto ``sys.path`` itself.
  * ``from dispatcher.rate_limiter import ...`` — used by tests/test_phase4.py.

The second style needs the repository root on ``sys.path``. ``python -m
pytest`` adds it implicitly (the current working directory), but bare
``pytest tests/`` — which is what CI runs — only prepends ``tests/``, so
collection of tests/test_phase4.py fails with ModuleNotFoundError. Importing
this file happens before any test module, so the root is always available.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

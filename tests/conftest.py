"""Pytest sys.path setup.

The package (``polar_high``) is installed via ``pip install -e .``;
tests import it as a regular dependency.

``tests/fixtures/`` holds small toy models used by engine tests and is
added to ``sys.path`` so test files can do ``from toy_data import ...``
etc.  Also configured via ``pyproject.toml``'s ``pythonpath`` — the
conftest mirrors it for editor / standalone invocations.
"""

import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

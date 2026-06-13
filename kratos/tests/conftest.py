"""Make the ``kratos`` package importable when pytest is run from anywhere.

Adds the directory that CONTAINS the ``kratos`` package (i.e. the parent of
this package) to sys.path, so ``import kratos.*`` resolves to the live source
tree being tested.
"""

import sys
from pathlib import Path

# tests/  ->  kratos/  ->  <parent that holds the package>
_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

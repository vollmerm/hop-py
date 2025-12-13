import os
import sys

# Ensure tests can import top-level modules when pytest changes CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

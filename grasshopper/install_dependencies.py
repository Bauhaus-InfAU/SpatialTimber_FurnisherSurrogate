#! python 3
"""Install furnisher surrogate dependencies into Rhino 8's CPython.

Open this file in Rhino's Script Editor and click the green Run button.
It will install PyTorch (CPU-only) and the furnisher_surrogate package.
Packages that are already installed will be skipped.

This only needs to run once per Rhino installation.
"""

import os
import site
import sys
from pip._internal.cli.main import main as pip_main

# Locate the package root (parent of the grasshopper/ directory this script lives in)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(_script_dir)
_src_dir = os.path.join(_package_root, "src")


def pip_install(*args):
    """Run pip install in-process (no subprocess needed)."""
    cmd = ["install", *args]
    print(f"pip install {' '.join(args)}")
    pip_main(cmd)
    print()


# 1. PyTorch CPU-only (~200 MB on first install, skipped if present)
print("=" * 60)
print("Step 1/3: Installing PyTorch (CPU-only)...")
print("=" * 60)
pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

# 2. numpy + Pillow (inference deps, installed via pip — no subprocess needed)
print("=" * 60)
print("Step 2/3: Installing numpy and Pillow...")
print("=" * 60)
pip_install("numpy", "Pillow")

# 3. furnisher_surrogate — add the local src/ directory via a .pth file.
#    This avoids running pip on the local package, which would trigger hatchling's
#    build backend as a .py subprocess.  On Windows, .py files may be associated
#    with Rhino, causing a new Rhino window to open instead of running Python.
print("=" * 60)
print("Step 3/3: Registering furnisher_surrogate from local repo...")
print(f"  Source: {_src_dir}")
print("=" * 60)
_site_packages = site.getsitepackages()[0]
_pth_path = os.path.join(_site_packages, "furnisher_surrogate_local.pth")
with open(_pth_path, "w") as _f:
    _f.write(_src_dir + "\n")
# Also make it importable in the current session without a restart
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
print(f"Written: {_pth_path}")
print()

# 4. Verify
print()
print("=" * 60)
print("Verifying installation...")
print("=" * 60)
try:
    from furnisher_surrogate.predict import predict_score  # noqa: F401
    print("furnisher_surrogate ... OK")
except ImportError as e:
    print(f"furnisher_surrogate ... FAILED: {e}")

try:
    import torch  # noqa: F401
    print(f"torch {torch.__version__} ... OK")
except ImportError as e:
    print(f"torch ... FAILED: {e}")

try:
    import numpy as np  # noqa: F401
    print(f"numpy {np.__version__} ... OK")
except ImportError as e:
    print(f"numpy ... FAILED: {e}")

print()
print("Done. You can close this script.")

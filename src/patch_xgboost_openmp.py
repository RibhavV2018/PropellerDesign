"""Make XGBoost importable on macOS without Homebrew.

THE PROBLEM

XGBoost's macOS wheel needs an OpenMP runtime, and its libxgboost.dylib looks
for it at a hardcoded Homebrew rpath:

    @rpath/libomp.dylib  ->  /opt/homebrew/opt/libomp/lib/libomp.dylib

On a machine without Homebrew that path does not exist, so `import xgboost`
fails with an unhelpful "Library not loaded" error. The usual advice is
`brew install libomp`, which means installing a package manager to satisfy one
dynamic library.

THE FIX

scikit-learn's wheel already ships its own libomp.dylib. This copies it next
to libxgboost.dylib and rewrites the load path to @loader_path, so XGBoost
finds the copy sitting beside it. No sudo, nothing written outside the venv,
and no system paths polluted.

Run once after creating the venv, or after any `pip install`/upgrade of
xgboost, which restores the original binary:

    python src/patch_xgboost_openmp.py

Idempotent, and a no-op on Linux and Windows, where the wheels bundle their
own runtime.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin":
        print("not macOS — nothing to do")
        return 0

    try:
        import xgboost  # noqa: F401
        print("xgboost already imports cleanly — nothing to do")
        return 0
    except Exception:
        pass

    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    src = site / "sklearn" / ".dylibs" / "libomp.dylib"
    lib = site / "xgboost" / "lib"
    dylib = lib / "libxgboost.dylib"

    if not src.exists():
        print(f"no libomp found at {src}\n  install scikit-learn first: pip install scikit-learn")
        return 1
    if not dylib.exists():
        print(f"no libxgboost.dylib at {dylib}\n  install xgboost first: pip install xgboost")
        return 1

    shutil.copy2(src, lib / "libomp.dylib")
    subprocess.run(
        ["install_name_tool", "-change", "@rpath/libomp.dylib",
         "@loader_path/libomp.dylib", str(dylib)],
        check=True,
    )
    print(f"copied {src.name} into {lib}")
    print("rewrote libxgboost.dylib load path to @loader_path")

    import importlib
    importlib.invalidate_caches()
    try:
        import xgboost
        print(f"xgboost {xgboost.__version__} now imports")
        return 0
    except Exception as exc:
        print(f"still failing: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

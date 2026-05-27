#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check dependencies for the French circuit-theory PDF-to-Markdown pipeline.
"""

import sys
import subprocess


def check_package(package_name, import_name=None):
    """Check if a package is installed by trying to import it."""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        print(f"[OK] {package_name} is installed")
        return True
    except ImportError:
        print(f"[MISSING] {package_name} is NOT installed")
        return False


def main():
    print("Checking dependencies for the pipeline...\n")

    # Core dependencies from the code
    dependencies = [
        ("torch", "torch"),
        ("Pillow", "PIL"),
        ("PyMuPDF", "fitz"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
    ]

    all_installed = True
    for pkg, imp in dependencies:
        if not check_package(pkg, imp):
            all_installed = False

    print("\n" + "=" * 50)
    if all_installed:
        print("[PASS] All dependencies are installed!")
        print("You can run the pipeline.")
    else:
        print("[FAIL] Some dependencies are missing.")
        print("Please install them using:")
        print("  pip install torch Pillow PyMuPDF transformers numpy")
        print("\nNote: For torch, you might need the CUDA version if you have a GPU.")
        print(
            "      Visit https://pytorch.org/get-started/locally/ for the correct command."
        )

    return 0 if all_installed else 1


if __name__ == "__main__":
    sys.exit(main())

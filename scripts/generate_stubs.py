#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main():
    # The module to generate stubs for
    module = "pyftle.ginterp"

    # Output directory for stubs: pybind11_stubgen appends pyftle at the end
    output_dir = Path(__file__).parent.parent / "src"

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pybind11-stubgen
    subprocess.run(
        [sys.executable, "-m", "pybind11_stubgen", module, "-o", str(output_dir)],
        check=True,
    )

    # Move the stubs from the 'pyftle' subdirectory to the correct location
    target_dir = Path(__file__).parent.parent / "src" / "pyftle"

    print(f"Stubs generated for {module} in {target_dir}")


if __name__ == "__main__":
    main()

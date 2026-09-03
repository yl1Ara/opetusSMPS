#!/usr/bin/env python3
"""Compile Python sources in memory without creating deployment artifacts."""

import sys
import tokenize
from pathlib import Path


def sources(path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from path.rglob("*.py")
    else:
        raise FileNotFoundError(path)


def main():
    checked = 0
    for argument in sys.argv[1:]:
        for path in sources(Path(argument)):
            with tokenize.open(path) as source_file:
                compile(source_file.read(), str(path), "exec")
            checked += 1
    if checked == 0:
        raise RuntimeError("No Python sources were checked")
    print(f"Checked {checked} Python source files")


if __name__ == "__main__":
    main()

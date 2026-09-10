#!/usr/bin/env python3
"""Measure CLI selection printing, including startup, parsing, lowering, and I/O.

Build first: cargo build --release -p zirium --bin zirium
Run with an optional binary path to compare two release builds.
Each size uses one warmup and the median of three measured subprocesses.
"""

import argparse
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "binary",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "target/release/zirium",
    )
    binary = parser.parse_args().binary.resolve()
    for size in (128, 512, 1024):
        source = (
            "module {\n"
            + "".join(f"%c{i} = arith.constant {i} : i32\n" for i in range(size))
            + "}\n"
        ).encode()
        samples = []
        for run in range(4):
            start = time.perf_counter()
            result = subprocess.run(
                [str(binary), "--strict", ""],
                input=source,
                capture_output=True,
                timeout=60,
                check=True,
            )
            elapsed = time.perf_counter() - start
            assert result.stdout.count(b"arith.constant") == size
            assert not result.stderr
            if run:
                samples.append(elapsed)
        print(
            f"constants={size} median_ms={statistics.median(samples) * 1000:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

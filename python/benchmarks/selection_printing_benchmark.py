#!/usr/bin/env python3
"""Measure CLI selection printing, including startup, parsing, lowering, and I/O.

Build first: cargo build --release -p zirium --bin zirium
Run with an optional binary path to compare two release builds.
The default workload uses one warmup and the median of three measured subprocesses.

Use --output-rss for an opt-in output-amplification run. That mode launches each
measurement through a fresh helper process, redirects CLI stdout to a file, and
checks every emitted byte without retaining the amplified output in the helper.
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def selection_source(size):
    return (
        "module {\n"
        + "".join(f"%c{i} = arith.constant {i} : i32\n" for i in range(size))
        + "}\n"
    ).encode()


def peak_child_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def rustc_version():
    try:
        return subprocess.run(
            ["rustc", "-V"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def verify_repeated_output(output_path, expected_path, repetitions):
    expected = expected_path.read_bytes()
    with output_path.open("rb") as output:
        for _ in range(repetitions):
            if output.read(len(expected)) != expected:
                raise RuntimeError("CLI output differs from the one-emission reference")
        if output.read(1):
            raise RuntimeError("CLI output contains bytes after the expected emissions")
    return len(expected) * repetitions


def run_output_rss_child(args):
    query = "emit; " * (args.child_emissions - 1) + "emit"
    cli = [str(args.binary)]
    if args.child_mode == "jsonl":
        cli.append("--jsonl")
    cli.extend([query, str(args.child_input)])

    with tempfile.TemporaryDirectory(prefix="zirium-output-rss-") as directory:
        output_path = Path(directory) / "stdout"
        start = time.perf_counter()
        with output_path.open("wb") as output:
            result = subprocess.run(
                cli,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        elapsed = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        if result.stderr:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        emitted_bytes = verify_repeated_output(
            output_path, args.child_expected, args.child_emissions
        )
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "emitted_bytes": emitted_bytes,
                "peak_rss_bytes": peak_child_rss_bytes(),
            }
        )
    )


def run_retention_rss_child(args):
    cli = [str(args.binary)]
    if args.child_mode == "large-jsonl":
        cli.append("--jsonl")
        query = ""
    else:
        query = "count"
    cli.extend([query, str(args.child_input)])

    with tempfile.TemporaryDirectory(prefix="zirium-retention-rss-") as directory:
        output_path = Path(directory) / "stdout"
        start = time.perf_counter()
        with output_path.open("wb") as output:
            result = subprocess.run(
                cli,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        elapsed = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        if result.stderr:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        output = output_path.read_bytes()
        expected = args.child_expected.read_bytes()
        if output != expected:
            raise RuntimeError(f"{args.child_mode} output differs from exact reference")
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "emitted_bytes": len(output),
                "peak_rss_bytes": peak_child_rss_bytes(),
            }
        )
    )


def child_samples(command, warmups, runs):
    samples = []
    for index in range(warmups + runs):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        if index >= warmups:
            samples.append(json.loads(result.stdout))
    return samples


def run_retention_rss(binary, constants, warmups, runs):
    if binary.parent.name != "release":
        raise SystemExit("--retention-rss requires a binary from a release directory")

    with tempfile.TemporaryDirectory(prefix="zirium-retention-rss-input-") as directory:
        directory = Path(directory)
        input_path = directory / "input.mlir"
        source = selection_source(constants)
        input_path.write_bytes(source)

        normal = subprocess.run(
            [str(binary), "", str(input_path)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        if normal.stderr:
            raise RuntimeError(normal.stderr.decode(errors="replace"))
        expected = {
            "scalar-count": f"{constants + 1}\n".encode(),
            "large-jsonl": (
                json.dumps(
                    {
                        "document": str(input_path),
                        "result": normal.stdout.decode(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
        }

        for mode in ("scalar-count", "large-jsonl"):
            expected_path = directory / f"expected-{mode}"
            expected_path.write_bytes(expected[mode])
            samples = child_samples(
                [
                    sys.executable,
                    __file__,
                    str(binary),
                    "--_retention-rss-child",
                    "--child-mode",
                    mode,
                    "--child-input",
                    str(input_path),
                    "--child-expected",
                    str(expected_path),
                ],
                warmups,
                runs,
            )
            elapsed_ms = [sample["elapsed_seconds"] * 1000 for sample in samples]
            peak_mib = [sample["peak_rss_bytes"] / (1024 * 1024) for sample in samples]
            emitted_bytes = len(expected[mode])
            assert all(sample["emitted_bytes"] == emitted_bytes for sample in samples)
            print(
                " ".join(
                    [
                        f"platform={platform.platform()}",
                        f"python={platform.python_version()}",
                        f"rustc={json.dumps(rustc_version())}",
                        f"cpu_count={os.cpu_count()}",
                        "profile=release",
                        f"binary={binary}",
                        f"workload={mode}",
                        f"constants={constants}",
                        f"input_bytes={len(source)}",
                        f"output_bytes={emitted_bytes}",
                        f"warmups={warmups}",
                        f"measured_runs={runs}",
                        "boundary=fresh_process_startup_parse_lower_select_stage_write",
                        f"median_ms={statistics.median(elapsed_ms):.3f}",
                        f"spread_ms={max(elapsed_ms) - min(elapsed_ms):.3f}",
                        f"median_peak_rss_mib={statistics.median(peak_mib):.3f}",
                        f"spread_peak_rss_mib={max(peak_mib) - min(peak_mib):.3f}",
                    ]
                ),
                flush=True,
            )


def run_output_rss(binary, constants, emission_counts, warmups, runs):
    if binary.parent.name != "release":
        raise SystemExit("--output-rss requires a binary from a release directory")

    with tempfile.TemporaryDirectory(prefix="zirium-output-rss-input-") as directory:
        directory = Path(directory)
        input_path = directory / "input.mlir"
        source = selection_source(constants)
        input_path.write_bytes(source)
        for mode in ("normal", "jsonl"):
            expected_path = directory / f"expected-{mode}"
            command = [str(binary)]
            if mode == "jsonl":
                command.append("--jsonl")
            command.extend(["emit", str(input_path)])
            expected = subprocess.run(
                command, capture_output=True, timeout=60, check=True
            )
            if expected.stderr:
                raise RuntimeError(expected.stderr.decode(errors="replace"))
            expected_path.write_bytes(expected.stdout)

            for emissions in emission_counts:
                samples = child_samples(
                    [
                        sys.executable,
                        __file__,
                        str(binary),
                        "--_output-rss-child",
                        "--child-mode",
                        mode,
                        "--child-input",
                        str(input_path),
                        "--child-expected",
                        str(expected_path),
                        "--child-emissions",
                        str(emissions),
                    ],
                    warmups,
                    runs,
                )
                elapsed_ms = [sample["elapsed_seconds"] * 1000 for sample in samples]
                peak_mib = [
                    sample["peak_rss_bytes"] / (1024 * 1024) for sample in samples
                ]
                emitted_bytes = samples[0]["emitted_bytes"]
                assert all(
                    sample["emitted_bytes"] == emitted_bytes for sample in samples
                )
                print(
                    " ".join(
                        [
                            f"platform={platform.platform()}",
                            f"python={platform.python_version()}",
                            f"rustc={json.dumps(rustc_version())}",
                            "profile=release",
                            f"binary={binary}",
                            f"mode={mode}",
                            f"constants={constants}",
                            f"input_bytes={len(source)}",
                            f"emissions={emissions}",
                            f"emitted_bytes={emitted_bytes}",
                            f"warmups={warmups}",
                            f"measured_runs={runs}",
                            "boundary=fresh_process_startup_parse_lower_select_stage_write",
                            f"median_ms={statistics.median(elapsed_ms):.3f}",
                            f"spread_ms={max(elapsed_ms) - min(elapsed_ms):.3f}",
                            f"median_peak_rss_mib={statistics.median(peak_mib):.3f}",
                            f"spread_peak_rss_mib={max(peak_mib) - min(peak_mib):.3f}",
                        ]
                    ),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "binary",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "target/release/zirium",
    )
    parser.add_argument(
        "--output-rss",
        action="store_true",
        help="run the opt-in release output-amplification RSS workload",
    )
    parser.add_argument(
        "--retention-rss",
        action="store_true",
        help="run isolated scalar-count and single-large-JSONL release workloads",
    )
    parser.add_argument("--retention-constants", type=int, default=100_000)
    parser.add_argument("--retention-warmups", type=int, default=1)
    parser.add_argument("--retention-runs", type=int, default=5)
    parser.add_argument("--rss-constants", type=int, default=128)
    parser.add_argument(
        "--rss-emissions", type=int, nargs="+", default=[256, 1024, 4096]
    )
    parser.add_argument("--rss-runs", type=int, default=3)
    parser.add_argument("--rss-warmups", type=int, default=1)
    parser.add_argument(
        "--_output-rss-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_retention-rss-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--child-mode",
        choices=("normal", "jsonl", "scalar-count", "large-jsonl"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-expected", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-emissions", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if args._output_rss_child:
        run_output_rss_child(args)
        return
    if args._retention_rss_child:
        run_retention_rss_child(args)
        return
    if args.output_rss:
        if (
            args.rss_constants < 1
            or args.rss_warmups < 0
            or args.rss_runs < 1
            or any(emissions < 1 for emissions in args.rss_emissions)
        ):
            parser.error("RSS sizes/runs must be positive and warmups non-negative")
        run_output_rss(
            args.binary,
            args.rss_constants,
            args.rss_emissions,
            args.rss_warmups,
            args.rss_runs,
        )
        return
    if args.retention_rss:
        if (
            args.retention_constants < 1
            or args.retention_warmups < 0
            or args.retention_runs < 1
        ):
            parser.error(
                "retention sizes/runs must be positive and warmups non-negative"
            )
        run_retention_rss(
            args.binary,
            args.retention_constants,
            args.retention_warmups,
            args.retention_runs,
        )
        return

    binary = args.binary
    for size in (128, 512, 1024):
        source = selection_source(size)
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

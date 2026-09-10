#!/usr/bin/env python3
"""CLI review probes and generated StableHLO DAG checks (standard library only).

Build first: cargo build --release -p zirium --bin zirium
Run: python3 python/benchmarks/query_language_review.py > target/query-review/results.json
Timings include process startup, parsing, lowering, evaluation, and output.
Known limitations are recorded as observations; corrected failures have explicit checks.
"""

import json
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target/release/zirium"
REGISTRY = ROOT / "crates/zirium/registries/stablehlo.json"
results = []


def run(
    label,
    query,
    source,
    *,
    registry=True,
    expected=None,
    expected_error=None,
    timeout=10,
):
    args = [str(BIN)]
    if registry:
        args += ["--registry", str(REGISTRY)]
    # A file also permits long queries beyond the OS per-argument limit.
    with tempfile.TemporaryDirectory(prefix="zirium-query-review-") as directory:
        program = Path(directory) / "query.zirium"
        program.write_text(query)
        start = time.perf_counter()
        try:
            p = subprocess.run(
                [*args, "-f", str(program)],
                input=source,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            row = {
                "case": label,
                "seconds": time.perf_counter() - start,
                "code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except subprocess.TimeoutExpired:
            row = {
                "case": label,
                "seconds": time.perf_counter() - start,
                "timeout": True,
            }
    if expected is not None:
        assert row.get("code") == 0 and row["stdout"] == expected, row
        row["checked"] = True
    if expected_error is not None:
        stderr = row.get("stderr")
        assert (
            row.get("code") == 1
            and not row["stdout"]
            and isinstance(stderr, str)
            and expected_error in stderr
        ), row
        row["checked"] = True
    results.append(row)
    return row


def dag(seed, n):
    rng = random.Random(seed)
    edges = {0: []}
    lines = [
        "module {",
        "func.func @dag() -> tensor<f32> {",
        '%v0 = "stablehlo.constant"() {value = dense<1.0> : tensor<f32>, review.id = "0"} : () -> tensor<f32>',
    ]
    for i in range(1, n):
        edges[i] = [rng.randrange(i), rng.randrange(i)]
        a, b = edges[i]
        lines.append(
            f'%v{i} = "stablehlo.add"(%v{a}, %v{b}) {{review.id = "{i}"}} : (tensor<f32>, tensor<f32>) -> tensor<f32>'
        )
    lines += [f"func.return %v{n - 1} : tensor<f32>", "}", "}"]
    return "\n".join(lines), edges


def expected_values(values):
    # json pretty output is compared semantically below.
    return [str(i) for i in values]


def check_graph(seed, n=60):
    source, edges = dag(seed, n)
    selected = list(range(0, n, 7))
    pred = " or ".join(f'string_attr_eq("review.id", "{i}")' for i in selected)
    prefix = f"filter({pred})"
    defs = [j for i in selected for j in edges[i]]
    users = [
        j for i in selected for j in range(n) for operand in edges[j] if operand == i
    ]
    ancestors = set(selected)
    while True:
        expanded = ancestors | {j for i in ancestors for j in edges[i]}
        if expanded == ancestors:
            break
        ancestors = expanded
    descendants = set(selected)
    while True:
        expanded = descendants | {
            i for i in range(n) if any(j in descendants for j in edges[i])
        }
        if expanded == descendants:
            break
        descendants = expanded
    cases = [
        ("defs", defs),
        ("users", users),
        ("defs | unique", list(dict.fromkeys(defs))),
        ("(defs union users)", sorted(set(defs) | set(users))),
        ("(defs intersect users)", sorted(set(defs) & set(users))),
        ("(defs except users)", sorted(set(defs) - set(users))),
        ("fixpoint(closure)", sorted(ancestors)),
        ("slice", sorted(ancestors)),
        ("fixpoint(filter(true) union users)", sorted(descendants)),
    ]
    for stage, expected in cases:
        row = run(
            f"dag-{seed}-{stage}",
            f'{prefix} | {stage} | attr("review.id") | json',
            source,
        )
        assert row.get("code") == 0, row
        assert json.loads(row["stdout"]) == expected_values(expected), row
        row["checked"] = True


def chain(n, argument=False):
    header = "%arg: tensor<f32>" if argument else ""
    lines = [
        "module {",
        f"func.func @chain({header}) -> tensor<f32> {{",
        "%v0 = stablehlo.constant dense<1.0> : tensor<f32>",
    ]
    for i in range(1, n + 1):
        rhs = "%arg" if argument else "%v0"
        lines.append(f"%v{i} = stablehlo.add %v{i - 1}, {rhs} : tensor<f32>")
    lines += [f"func.return %v{n} : tensor<f32>", "}", "}"]
    return "\n".join(lines)


def semantic_probes(decoder):
    run(
        "decoder-tag",
        'filter(op("stablehlo.dot_general")) | set_attr("review.tag", "matmul") | count',
        decoder,
        expected="19\n",
    )
    run(
        "decoder-contracting-dims",
        'filter(op("stablehlo.dot_general")) | attr("contracting_dims") | unique | json',
        decoder,
    )
    generic_return = decoder.replace(
        "return %233, %169, %171 : tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>",
        '"func.return"(%233, %169, %171) : (tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>) -> ()',
    )
    run(
        "decoder-generic-return-closure",
        'filter(op("stablehlo.dot_general")) | fixpoint(closure) | count',
        generic_return,
        expected="236\n",
    )
    run(
        "decoder-generic-return-tag",
        'filter(op("stablehlo.dot_general")) | set_attr("review.tag", "matmul") | count',
        generic_return,
        expected="19\n",
    )
    explicit = """module {
      func.func @reduce(%x: tensor<4xf32>) -> tensor<f32> {
        %c = stablehlo.constant dense<0.0> : tensor<f32>
        %r = "stablehlo.reduce"(%x, %c) ({
        ^bb0(%a: tensor<f32>, %b: tensor<f32>):
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }) {dimensions = array<i64: 0>} : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
        func.return %r : tensor<f32>
      }
    }"""
    run("explicit-reduce-count", "count", explicit, expected="7\n")
    run(
        "explicit-reduce-children",
        'filter(op("stablehlo.reduce")) | children | count',
        explicit,
        expected="2\n",
    )
    run(
        "explicit-reduce-closure",
        'filter(op("stablehlo.reduce")) | fixpoint(closure) | count',
        explicit,
        expected="6\n",
    )
    # The same arithmetic reduction in compact assembly has no exposed body.
    compact = """module {
      func.func @reduce(%x: tensor<4xf32>) -> tensor<f32> {
        %c = stablehlo.constant dense<0.0> : tensor<f32>
        %r = stablehlo.reduce(%x init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
        func.return %r : tensor<f32>
      }
    }"""
    run("compact-reduce-count", "count", compact)
    run(
        "compact-reduce-children",
        'filter(op("stablehlo.reduce")) | children | count',
        compact,
    )


def aggregation_graph_probes():
    # An independent graph oracle: every function has a random number of adds,
    # and calls may share targets, point backwards, or form recursive cycles.
    for seed in range(12):
        rng = random.Random(seed)
        n = 30
        edges = {
            i: [rng.randrange(n) for _ in range(rng.randrange(4))] for i in range(n)
        }
        adds = {i: rng.randrange(5) for i in range(n)}
        lines = ["module {"]
        for i in range(n):
            lines.append(f"func.func @f{i}(%arg: tensor<f32>) {{")
            for j in range(adds[i]):
                lines.append(f"%v{j} = stablehlo.add %arg, %arg : tensor<f32>")
            for target in edges[i]:
                lines.append(f"func.call @f{target}(%arg) : (tensor<f32>) -> ()")
            lines += ["func.return", "}"]
        lines.append("}")
        source = "\n".join(lines)
        for transitive in [False, True]:
            expected = {}
            for i in range(n):
                visited = {i}
                pending = [i] if transitive else []
                while pending:
                    for target in edges[pending.pop()]:
                        if target not in visited:
                            visited.add(target)
                            pending.append(target)
                histogram = {
                    "stablehlo.add": sum(adds[j] for j in visited),
                    "func.call": sum(len(edges[j]) for j in visited),
                    "func.return": len(visited),
                }
                expected[f"f{i}"] = {
                    key: value for key, value in histogram.items() if value
                }
            traversal = "reachable | " if transitive else ""
            row = run(
                f"function-graph-{seed}-reachable-{transitive}",
                'F = filter(op("func.func")); F | map_by(attr("sym_name"), '
                f"children | subtree | {traversal}names | tally) | json",
                source,
            )
            assert row.get("code") == 0, row
            assert json.loads(row["stdout"]) == expected, row
            row["checked"] = True

    for n in [128, 512, 2048, 8192]:
        source = (
            "module {\n"
            + "\n".join(
                f'func.func @f{i}() attributes {{entry = "{i}"}} {{ '
                + (f"func.call @f{i + 1}() : () -> () " if i + 1 < n else "")
                + "func.return }"
                for i in range(n)
            )
            + "\n}"
        )
        row = run(
            f"reachable-call-chain-{n}",
            'F = filter(string_attr_eq("entry", "0")); '
            'F | map_by(attr("sym_name"), children | reachable | names | tally)',
            source,
        )
        assert row.get("code") == 0, row
        assert json.loads(row["stdout"]) == {
            "f0": {"func.call": n - 1, "func.return": n}
        }, row
        row["checked"] = True

    run(
        "many-saved-bindings",
        'A0 = filter(op("stablehlo.add"));'
        + "".join(f"A{i} = A{i - 1};" for i in range(1, 1000))
        + "A999 | count",
        dag(0, 60)[0],
        expected="59\n",
    )


def markdown_report_probes():
    for n in [1, 32, 256, 1024]:
        lines = ["module {"]
        for i in range(n):
            lines.append(f"func.func @f{i}(%arg: tensor<f32>) {{")
            for j in range(i % 3 + 1):
                lines.append(f"%v{j} = stablehlo.add %arg, %arg : tensor<f32>")
            lines += ["func.return", "}"]
        lines.append("}")
        expected = (
            f"Functions: {n}\n\n"
            "| Key | func.return | stablehlo.add |\n"
            "| --- | --- | --- |\n"
            + "".join(
                f"| f{i} | 1 | {i % 3 + 1} |\n"
                for i in sorted(range(n), key=lambda i: f"f{i}")
            )
            + "\nDone.\n"
        )
        run(
            f"markdown-report-{n}",
            'F = filter(op("func.func")); N = F | count; '
            'print("Functions: {N}"); '
            'F | map_by(attr("sym_name"), children | names | tally) | markdown; '
            'print("Done.");',
            "\n".join(lines),
            expected=expected,
        )


def json_literal_probes():
    source, _ = dag(0, 60)
    counts = {
        "builtin.module": 1,
        "func.func": 1,
        "func.return": 1,
        "stablehlo.constant": 1,
        "stablehlo.add": 59,
    }
    for n in [16, 256, 2048]:
        query = (
            'N = filter(op("stablehlo.add")) | count; Counts = names | tally; '
            '{"title": "{N} additions", "sections": ['
            + ",".join(f'{{"index": {i}, "counts": Counts}}' for i in range(n))
            + "]}"
        )
        row = run(f"json-literal-{n}-sections", query, source)
        assert row.get("code") == 0, row
        assert json.loads(row["stdout"]) == {
            "title": "59 additions",
            "sections": [{"index": i, "counts": counts} for i in range(n)],
        }, row
        row["checked"] = True


def main():
    decoder = (ROOT / "examples/cli/stablelm-decode.mlir").read_text()
    semantic_probes(decoder)
    queries = {
        "all": "count",
        "matmuls": 'filter(op("stablehlo.dot_general")) | count',
        "reductions": 'filter(op("stablehlo.reduce")) | count',
        "matmul-defs": 'filter(op("stablehlo.dot_general")) | defs | unique | count',
        "matmul-users": 'filter(op("stablehlo.dot_general")) | users | unique | count',
        "matmul-functions": 'filter(op("stablehlo.dot_general")) | root(op("func.func")) | unique | attr("sym_name") | json',
        "broadcast-dims": 'filter(op("stablehlo.broadcast_in_dim") and has_attr("dims")) | count',
        "return": 'filter(op("func.return")) | count',
        "closure": 'filter(op("stablehlo.dot_general")) | fixpoint(closure) | count',
        "reduce-children": 'filter(op("stablehlo.reduce")) | children | count',
        "downstream": 'filter(op("stablehlo.dot_general")) | fixpoint(filter(true) union users) | count',
        "cache-updates": 'filter(op("stablehlo.dynamic_update_slice")) | count',
    }
    for label, query in queries.items():
        for registry in [False, True]:
            run(
                f"decoder-{label}-registry={registry}",
                query,
                decoder,
                registry=registry,
            )
    run(
        "decoder-qualified-return-closure",
        queries["closure"],
        decoder.replace("\n    return ", "\n    func.return "),
        expected="236\n",
    )
    for seed in range(12):
        check_graph(seed)
    for n in [128, 512, 2048, 8192]:
        source = chain(n)
        for label, query, expected in [
            ("count", "count", n + 4),
            ("slice", 'filter(op("func.return")) | fixpoint(closure) | count', n + 2),
        ]:
            run(f"chain-{n}-{label}-warmup", query, source, expected=f"{expected}\n")
            rows = [
                run(f"chain-{n}-{label}-{rep}", query, source, expected=f"{expected}\n")
                for rep in range(3)
            ]
            results.append(
                {
                    "case": f"chain-{n}-{label}-median",
                    "seconds": statistics.median(r["seconds"] for r in rows),
                }
            )
    for n in [128, 512, 2048, 8192]:
        run(
            f"argument-fanout-{n}",
            'filter(op("func.return")) | fixpoint(closure) | count',
            chain(n, True),
            expected=f"{n + 3}\n",
        )
        run(
            f"argument-slice-{n}",
            'filter(op("func.return")) | slice | count',
            chain(n, True),
            expected=f"{n + 2}\n",
        )
    small = "module { %c = arith.constant 1 : i32 }"
    run(
        "nonconverging-subtree",
        "fixpoint(subtree) | count",
        small,
        registry=False,
        expected_error="work limit",
    )
    run(
        "converging-subtree",
        "fixpoint(subtree | unique) | count",
        small,
        registry=False,
        expected="2\n",
    )
    run(
        "cycle",
        'filter(op("builtin.module")) | fixpoint(children union parent) | count',
        small,
        registry=False,
        expected_error="cycles",
    )
    for n in [1000, 10000]:
        run(
            f"flat-pipeline-{n}",
            " | ".join(["filter(true)"] * n + ["count"]),
            small,
            registry=False,
            expected="2\n",
        )
        run(
            f"flat-boolean-{n}",
            "filter(" + " or ".join(["false"] * n + ["true"]) + ") | count",
            small,
            registry=False,
            expected="2\n",
        )
    run(
        "over-nested",
        "(" * 70 + "input" + ")" * 70,
        small,
        registry=False,
        expected_error="nesting limit",
    )
    run(
        "unknown-custom-attributes",
        'filter(op("vendor.thing")) | attr("tag") | json',
        'module { %x = vendor.thing {tag = "present"} : i32 }',
        registry=False,
    )
    run(
        "byte-string-projection",
        'attr("tag") | count',
        'module { "vendor.thing"() {tag = "\\FF"} : () -> () }',
        registry=False,
        expected_error="UTF-8",
    )
    run(
        "value-set-order",
        '(filter(string_attr_eq("review.id", "2")) | attr("review.id") union filter(string_attr_eq("review.id", "0")) | attr("review.id")) | json',
        dag(0, 3)[0],
    )
    aggregation_graph_probes()
    markdown_report_probes()
    json_literal_probes()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

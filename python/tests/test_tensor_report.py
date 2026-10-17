import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
REPORT = ROOT / "python/examples/tensor_report.py"


def test_tensor_report_sizes_unknowns_locations_and_top_limit(tmp_path: Path) -> None:
    source = """module {
  func.func @first() {
    %a, %b, %c, %d, %e, %f, %g = "example.results"() : () -> (tensor<2x3xf32>, tensor<?x3xf32>, tensor<*xf32>, tensor<i32>, tensor<0x4xi16>, tensor<2x!example.element>, tensor<8xi1>) loc("model.py":7:3)
    return
  }
  func.func @second() {
    %a = "example.next"() : () -> tensor<4xf32>
    return
  }
}"""
    path = tmp_path / "tensors.mlir"
    path.write_text(source)
    command = [sys.executable, str(REPORT), str(path)]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(output.stdout)
    assert report["tensor_results"] == 8
    assert report["unknown_byte_sizes"] == 4
    rows = report["results"]
    assert [row["logical_bytes"] for row in rows] == [
        24,
        16,
        4,
        0,
        None,
        None,
        None,
        None,
    ]
    first = rows[0]
    assert first["function"] == "first"
    assert first["operation"] == "example.results"
    assert first["operation_index"] == 2
    assert first["result_index"] == 0
    assert first["dimensions"] == [2, 3]
    assert first["element_type"] == "f32"
    assert first["elements"] == 6
    assert first["location"] == 'loc("model.py":7:3)'
    start, end = first["source_range"]
    assert source.encode()[start:end].startswith(b"%a, %b, %c")
    assert rows[1]["function"] == "second"
    assert rows[1]["location"] is None
    by_result = {row["result_index"]: row for row in rows if row["function"] == "first"}
    assert by_result[1]["dimensions"] == [None, 3]
    assert by_result[1]["elements"] is None
    assert by_result[2]["dimensions"] is None
    assert by_result[2]["elements"] is None
    assert by_result[3]["dimensions"] == []
    assert by_result[3]["elements"] == 1
    assert by_result[4]["elements"] == 0
    assert by_result[5]["element_type"] == "!example.element"
    assert by_result[5]["elements"] == 2
    assert by_result[6]["elements"] == 8

    limited = subprocess.run(
        command + ["--top", "2"], check=True, capture_output=True, text=True
    )
    assert json.loads(limited.stdout)["results"] == rows[:2]

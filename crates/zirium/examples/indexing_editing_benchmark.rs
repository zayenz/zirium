use std::{hint::black_box, sync::Arc, time::Instant};

use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    semantic::{LoweringMode, lower_with_dialect_registry},
};

const OPERATIONS: usize = 2_000;

fn main() {
    let constants = OPERATIONS / 2;
    let mut source = String::from("builtin.module { func.func @bench() -> i32 {\n");
    for index in 0..constants {
        source.push_str(&format!("%v{index} = arith.constant {index} : i32\n"));
    }
    for index in constants..OPERATIONS {
        source.push_str(&format!(
            "%v{index} = arith.addi %v{}, %v0 : i32\n",
            index - 1
        ));
    }
    source.push_str(&format!("func.return %v{} : i32\n}} }}", OPERATIONS - 1));
    let start = Instant::now();
    let parsed = ParsedFile::parse_with_registry(
        Arc::<[u8]>::from(source.into_bytes()),
        DialectRegistry::proving(),
    )
    .unwrap();
    let parse_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let mut document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
            .document
            .unwrap();
    let lower_ns = start.elapsed().as_nanos();
    let parse_only = document.statistics();

    let operations = document.operations().collect::<Vec<_>>();
    let first = operations
        .iter()
        .copied()
        .find(|operation| document.operation_name(*operation) == Some("arith.constant"))
        .unwrap();
    let first_value = document.operation(first).unwrap().result(first, 0).unwrap();
    let start = Instant::now();
    black_box(document.uses(first_value));
    let use_index_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    black_box(document.symbol_index_diagnostics(DialectRegistry::proving()));
    let symbol_index_ns = start.elapsed().as_nanos();
    let last = operations
        .iter()
        .copied()
        .rev()
        .find(|operation| document.operation_name(*operation) == Some("arith.addi"))
        .unwrap();
    let start = Instant::now();
    black_box(document.dominates(first_value, last, DialectRegistry::proving()));
    let dominance_index_ns = start.elapsed().as_nanos();
    let indexed = document.statistics();
    let second = operations
        .iter()
        .copied()
        .filter(|operation| document.operation_name(*operation) == Some("arith.constant"))
        .nth(1)
        .unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    let second_value = editor
        .document()
        .operation(second)
        .unwrap()
        .result(second, 0)
        .unwrap();
    for value in [second_value, first_value].into_iter().cycle().take(20) {
        editor.rewire_operand(last, 1, value).unwrap();
    }
    let before = editor.document().statistics().pooled_list_entries;
    let start = Instant::now();
    let reclaimed = editor.compact_pools();
    let compaction_ns = start.elapsed().as_nanos();
    editor.commit().unwrap();

    println!(
        "operations={OPERATIONS} parse_ns={parse_ns} lower_ns={lower_ns} parse_only_index_entries={} use_index_ns={use_index_ns} symbol_index_ns={symbol_index_ns} dominance_index_ns={dominance_index_ns} indexed_entries={} pooled_before={before} pooled_reclaimed={reclaimed} compaction_ns={compaction_ns}",
        parse_only.use_index_entries
            + parse_only.symbol_index_entries
            + parse_only.dominance_index_entries,
        indexed.use_index_entries + indexed.symbol_index_entries + indexed.dominance_index_entries,
    );
}

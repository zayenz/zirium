use std::{fs, process::Command};

fn fixture(name: &str, source: &str) -> std::path::PathBuf {
    let directory =
        std::env::temp_dir().join(format!("zirium-diff-test-{}-{}", std::process::id(), name));
    fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("{name}.mlir"));
    fs::write(&path, source).unwrap();
    path
}

fn pair() -> (std::path::PathBuf, std::path::PathBuf) {
    (
        fixture("before", "module { %x = arith.constant 4 : i32 }\n"),
        fixture("after", "module { %renamed = arith.constant 8 : i32 }\n"),
    )
}

#[test]
fn adjacent_diff_paths_support_default_and_filtered_json_reports() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("1 modified"), "{text}");
    assert!(text.contains("attributes.value: 4 -> 8"), "{text}");

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"filter(changed("attributes") and dialect("arith")) | json"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
    assert_eq!(json[0]["kind"], "modified");
}

#[test]
fn validates_paired_input_grammar_before_reading() {
    for arguments in [
        vec!["--diff"],
        vec!["--diff", "before.mlir"],
        vec!["--diff", "-", "-"],
        vec!["--diff-locations"],
        vec!["--max-diff-work", "1"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
            .args(&arguments)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{arguments:?}");
        assert!(output.stdout.is_empty(), "{arguments:?}");
    }
}

#[test]
fn side_projection_prints_from_the_selected_document() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"filter(change("modified")) | before"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("constant 4"), "{text}");
    assert!(!text.contains("constant 8"), "{text}");
}

#[test]
fn projected_navigation_and_explicit_json_keep_diff_attribution() {
    let before = fixture(
        "navigation-before",
        "module { %x = arith.constant 4 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let after = fixture(
        "navigation-after",
        "module { %x = arith.constant 8 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--jsonl", "--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"filter(changed("attributes")) | after | users | unique | json"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["schema"], "zirium.diff.v1");
    assert_eq!(envelope["result_side"], "after");
    assert_eq!(envelope["comparison"]["locations"], "ignore");
    assert_eq!(envelope["comparison"]["opaque_values"], "bytes");
    assert_eq!(envelope["result"][0]["name"], "test.use");
}

#[test]
fn change_markdown_has_stable_context_columns() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg("markdown")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.starts_with("| Kind | Before context | After context | Changed fields |"),
        "{text}"
    );
    assert!(text.contains("| modified |"), "{text}");
    assert!(text.contains("attributes"), "{text}");
}

#[test]
fn statements_bindings_and_boolean_predicates_share_one_diff_session() {
    let before = fixture(
        "program-before",
        "module { %x = arith.constant 4 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let after = fixture(
        "program-after",
        "module { %x = arith.constant 8 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let program = r#"
        changed_constants = filter(change("added") or changed("attributes")) | filter(dialect("arith"));
        old_users = changed_constants | before | users | unique;
        new_users = changed_constants | after | users | unique;
        old_users | names;
        new_users | filter(op("test.use")) | names
    "#;
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(program)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "test.use\ntest.use\n"
    );
}

#[test]
fn diff_program_failure_is_atomic_and_mutations_are_rejected_before_loading() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg("count; before | before")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args([
            "--diff",
            "missing-before.mlir",
            "missing-after.mlir",
            "filter(true) | set_attr(\"x\", \"1\")",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("read-only"), "{error}");
    assert!(!error.contains("could not read before input"), "{error}");
}

#[test]
fn diff_query_work_limit_is_independent_from_matching_work() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--max-work", "1", "--diff"])
        .arg(&before)
        .arg(&after)
        .arg("filter(changed(\"attributes\")) | count")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("query work limit exceeded")
    );
}

#[test]
fn diff_sets_preserve_kind_and_reject_cross_side_empty_selections() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(
            "(filter(change(\"modified\")) | before union filter(change(\"added\")) | before) | count",
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"1\n");

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg("filter(false) | before union filter(false) | after")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("same diff side")
    );
}

#[test]
fn diff_scalar_projections_support_ordering_aggregation_and_markdown() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"filter(changed("attributes")) | result_types | sort | json"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::json!(["i32"])
    );

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"names | tally | markdown"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "| Key | Value |\n| --- | --- |\n| arith.constant | 1 |\n"
    );
}

#[test]
fn nested_diff_queries_keep_relative_input_and_map_across_sides() {
    let (before, after) = pair();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(r#"filter(changed("attributes")) | map_by(names, before | attr("value")) | json"#)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::json!({"arith.constant": ["4"]})
    );

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg("filter(false) | (count); filter(false) | (input | count)")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"0\n1\n");
}

#[test]
fn projected_diff_graph_traversal_reaches_unchanged_definitions() {
    let before = fixture(
        "graph-before",
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }\n",
    );
    let after = fixture(
        "graph-after",
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %y, %x : i32 }\n",
    );
    for traversal in ["slice", "closure", "reachable", "fixpoint(closure)"] {
        let query = format!("filter(changed(\"operands\")) | after | {traversal} | names");
        let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
            .args(["--diff"])
            .arg(&before)
            .arg(&after)
            .arg(query)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{traversal}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "arith.constant\narith.constant\narith.addi\n"
        );
    }
}

#[test]
fn strict_diff_reachable_rejects_unknown_reference_semantics() {
    let before = fixture(
        "strict-before",
        "module { %x = arith.constant 1 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let after = fixture(
        "strict-after",
        "module { %x = arith.constant 2 : i32 \"test.use\"(%x) : (i32) -> () }\n",
    );
    let query = "filter(changed(\"attributes\")) | after | users | reachable";
    let permissive = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(query)
        .output()
        .unwrap();
    assert!(permissive.status.success());

    let strict = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--strict", "--diff"])
        .arg(&before)
        .arg(&after)
        .arg(query)
        .output()
        .unwrap();
    assert!(!strict.status.success());
    assert!(strict.stdout.is_empty());
    assert!(
        String::from_utf8(strict.stderr)
            .unwrap()
            .contains("cannot determine reference semantics")
    );
}

#[test]
fn diff_records_can_be_ordered_by_relative_keys() {
    let before = fixture(
        "order-before",
        "module { \"test.z\"() {value = 1 : i64} : () -> () \"test.a\"() {value = 1 : i64} : () -> () }\n",
    );
    let after = fixture(
        "order-after",
        "module { \"test.z\"() {value = 2 : i64} : () -> () \"test.a\"() {value = 2 : i64} : () -> () }\n",
    );
    for (stage, expected) in [
        ("sort_by(names)", "test.a\ntest.z\n"),
        ("min_by(names)", "test.a\n"),
        ("max_all_by(names)", "test.z\n"),
    ] {
        let query = format!("filter(changed(\"attributes\")) | {stage} | names");
        let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
            .args(["--diff"])
            .arg(&before)
            .arg(&after)
            .arg(query)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{stage}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    }
}

#[test]
fn diff_print_interpolates_document_labels_and_scalar_bindings() {
    let (before, after) = pair();
    let program = "N = filter(changed(\"attributes\")) | count; print(\"{before_document} -> {after_document}: {N} change\")";
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg(program)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("{} -> {}: 1 change\n", before.display(), after.display())
    );

    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["--diff"])
        .arg(&before)
        .arg(&after)
        .arg("names | emit")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"arith.constant\n");
}

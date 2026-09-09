use std::{
    fs,
    io::{ErrorKind, Write},
    process::{Command, Stdio},
};

const INPUT: &str = "module {\n  // Initial value.\n  %c = arith.constant 7 : i32\n  // Double it.\n  %sum = arith.addi %c, %c : i32 // selected\n  \"example.observe\"(%sum) : (i32) -> ()\n}\n";

fn run_stdin(query: &str, input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg(query)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Err(error) = child.stdin.take().unwrap().write_all(input.as_bytes()) {
        assert_eq!(error.kind(), ErrorKind::BrokenPipe, "{error}");
    }
    child.wait_with_output().unwrap()
}

fn temporary_path(name: &str, extension: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "zirium-cli-{}-{name}.{extension}",
        std::process::id()
    ))
}

#[test]
fn stdin_selection_retains_shell_and_comments() {
    let output = run_stdin("filter(op(\"arith.addi\"))", INPUT);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("builtin.module"));
    assert!(text.contains("// Double it."));
    assert!(text.contains("arith.addi"));
    assert!(text.contains("// selected"));
    assert!(!text.contains("arith.constant"));
    assert!(!text.contains("example.observe"));
}

#[test]
fn boolean_predicates_select_names_and_decoded_string_attributes() {
    let input = "module {\n  \"test.a\"() {tag = \"say \\22hi\\22\"} : () -> ()\n  \"test.b\"() {tag = 7 : i32} : () -> ()\n  \"test.c\"() : () -> ()\n}\n";
    let output = run_stdin(
        r#"filter((op("test.a") or op("test.b")) and has_attr("tag") and not string_attr_eq("tag", "other"))"#,
        input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("test.a"), "{text}");
    assert!(text.contains("test.b"), "{text}");
    assert!(!text.contains("test.c"), "{text}");

    let decoded = run_stdin(r#"filter(string_attr_eq("tag", "say \"hi\""))"#, input);
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    let text = String::from_utf8(decoded.stdout).unwrap();
    assert!(text.contains("test.a"), "{text}");
    assert!(!text.contains("test.b"), "{text}");
}

#[test]
fn builtin_dense_array_attributes_support_queries_and_ownership() {
    let input = "module {\n  \"stablehlo.reduce\"() ({\n    \"stablehlo.add\"() : () -> ()\n  }) {dimensions = array<i64: 1>} : () -> ()\n}\n";
    for (query, expected) in [
        (
            r#"filter(op("stablehlo.reduce") and has_attr("dimensions")) | count"#,
            "1\n",
        ),
        (r#"filter(op("stablehlo.add")) | parent | count"#, "1\n"),
        (
            r#"filter(op("stablehlo.reduce")) | children | count"#,
            "1\n",
        ),
    ] {
        let output = run_stdin(query, input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    }
}

#[test]
fn malformed_predicates_produce_no_output() {
    for query in [
        r#"filter(string_attr_eq("tag" "value"))"#,
        r#"filter(op("arith.addi") and)"#,
        r#"filter(has_attr("bad name"))"#,
    ] {
        let output = run_stdin(query, INPUT);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("query error"));
    }
}

#[test]
fn closure_adds_shared_ssa_definition_once_with_comments() {
    let output = run_stdin("filter(op(\"arith.addi\")) | fixpoint(closure)", INPUT);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("arith.constant").count(), 1, "{text}");
    assert_eq!(text.matches("// Initial value.").count(), 1, "{text}");
    assert!(text.find("arith.constant").unwrap() < text.find("arith.addi").unwrap());
    assert!(!text.contains("example.observe"));
}

#[test]
fn direct_ssa_navigation_is_ordered_deduplicated_and_composable() {
    let defs = run_stdin("filter(op(\"arith.addi\")) | defs", INPUT);
    assert!(
        defs.status.success(),
        "{}",
        String::from_utf8_lossy(&defs.stderr)
    );
    let text = String::from_utf8(defs.stdout).unwrap();
    assert_eq!(text.matches("arith.constant").count(), 1, "{text}");
    assert!(!text.contains("arith.addi"), "{text}");

    let users = run_stdin("filter(op(\"arith.constant\")) | users | count", INPUT);
    assert!(
        users.status.success(),
        "{}",
        String::from_utf8_lossy(&users.stderr)
    );
    assert_eq!(String::from_utf8(users.stdout).unwrap(), "1\n");

    let unused = run_stdin("filter(op(\"example.observe\")) | users | count", INPUT);
    assert!(unused.status.success());
    assert_eq!(String::from_utf8(unused.stdout).unwrap(), "0\n");
}

#[test]
fn set_queries_are_ordered_and_composable() {
    let input = "module {\n  \"test.a\"() {group = \"keep\"} : () -> ()\n  \"test.b\"() {group = \"keep\"} : () -> ()\n  \"test.c\"() : () -> ()\n}\n";

    let union = run_stdin(
        r#"filter(op("test.c")) union filter(op("test.a") or has_attr("group"))"#,
        input,
    );
    assert!(
        union.status.success(),
        "{}",
        String::from_utf8_lossy(&union.stderr)
    );
    let text = String::from_utf8(union.stdout).unwrap();
    assert_eq!(text.matches("test.a").count(), 1, "{text}");
    assert_eq!(text.matches("test.b").count(), 1, "{text}");
    assert_eq!(text.matches("test.c").count(), 1, "{text}");
    assert!(
        text.find("test.a").unwrap() < text.find("test.c").unwrap(),
        "{text}"
    );

    let intersect = run_stdin(
        r#"(filter(has_attr("group")) intersect filter(not op("test.b"))) | count"#,
        input,
    );
    assert!(intersect.status.success());
    assert_eq!(String::from_utf8(intersect.stdout).unwrap(), "1\n");

    for query in [
        r#"(filter(has_attr("group")) except filter(string_attr_eq("group", "keep"))) | count"#,
        r#"(filter(op("missing")) union filter(op("missing"))) | count"#,
        r#"(filter(op("missing")) intersect filter(op("test.a"))) | count"#,
        r#"(filter(op("test.a")) except filter(op("missing"))) | count"#,
    ] {
        let output = run_stdin(query, input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = if query.contains("except filter(op") {
            "1\n"
        } else {
            "0\n"
        };
        assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    }

    let first = temporary_path("set-first", "mlir");
    let second = temporary_path("set-second", "mlir");
    fs::write(&first, input).unwrap();
    fs::write(&second, "module { \"test.d\"() : () -> () }\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg(r#"(filter(op("test.a")) union filter(op("test.d"))) | count"#)
        .arg(&first)
        .arg(&second)
        .output()
        .unwrap();
    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "1\n1\n");
}

#[test]
fn ownership_navigation_is_one_step_and_handles_block_arguments() {
    let input = "\"builtin.module\"() ({\n  \"func.func\"() ({\n  ^entry(%arg0: i32):\n    %sum = \"arith.addi\"(%arg0, %arg0) : (i32, i32) -> i32\n    \"func.return\"(%sum) : (i32) -> ()\n  }) : () -> ()\n}) : () -> ()\n";

    let boundary = run_stdin("filter(op(\"arith.addi\")) | defs", input);
    assert!(
        boundary.status.success(),
        "{}",
        String::from_utf8_lossy(&boundary.stderr)
    );
    let text = String::from_utf8(boundary.stdout).unwrap();
    assert!(text.contains("func.func"), "{text}");
    assert!(text.contains("arith.addi"), "{text}");

    let parent = run_stdin("filter(op(\"arith.addi\")) | parent", input);
    assert!(parent.status.success());
    let text = String::from_utf8(parent.stdout).unwrap();
    assert!(text.contains("func.func"), "{text}");

    let children = run_stdin("filter(op(\"func.func\")) | children", input);
    assert!(children.status.success());
    let text = String::from_utf8(children.stdout).unwrap();
    assert!(text.contains("arith.addi"), "{text}");
    assert!(text.contains("func.return"), "{text}");
}

#[test]
fn count_prints_one_scalar_line_per_input() {
    let first = temporary_path("count-first", "mlir");
    let second = temporary_path("count-second", "mlir");
    fs::write(&first, INPUT).unwrap();
    fs::write(&second, "module { func.return }\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("filter(op(\"arith.addi\")) | count")
        .arg(&first)
        .arg(&second)
        .output()
        .unwrap();
    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "1\n0\n");
}

#[test]
fn set_attr_keeps_the_changed_selection_and_comments() {
    let output = run_stdin(
        "filter(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\")",
        INPUT,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("analysis.tag = \"hot\"").count(), 1, "{text}");
    assert!(text.contains("// Double it."), "{text}");
    assert!(text.contains("// selected"), "{text}");
    assert!(!text.contains("arith.constant"), "{text}");
    assert!(!text.contains("example.observe"), "{text}");
}

#[test]
fn input_after_mutation_prints_the_whole_document() {
    let output = run_stdin(
        "filter(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | input | emit",
        INPUT,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("analysis.tag = \"hot\""), "{text}");
    assert!(text.contains("arith.constant"), "{text}");
    assert!(text.contains("example.observe"), "{text}");
}

#[test]
fn generic_single_operand_edit_prints_a_parseable_function_type() {
    let input =
        "%item = \"sample.create\"() : () -> i64\n\"sample.forward\"(%item) : (i64) -> ()\n";
    let edited = run_stdin(
        r#"filter(op("sample.forward")) | set_attr("roundtrip.marker", "present") | input | emit"#,
        input,
    );
    assert!(
        edited.status.success(),
        "{}",
        String::from_utf8_lossy(&edited.stderr)
    );
    let printed = String::from_utf8(edited.stdout).unwrap();
    assert!(
        printed.contains("roundtrip.marker = \"present\""),
        "{printed}"
    );

    let reparsed = run_stdin(r#"filter(op("sample.forward")) | count"#, &printed);
    assert!(
        reparsed.status.success(),
        "{printed}\n{}",
        String::from_utf8_lossy(&reparsed.stderr)
    );
    assert_eq!(reparsed.stdout, b"1\n", "{printed}");
}

#[test]
fn set_attr_mutates_the_selection_at_its_pipeline_position() {
    let output = run_stdin(
        "filter(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | fixpoint(closure) | input | emit",
        INPUT,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("analysis.tag = \"hot\"").count(), 1, "{text}");
    let constant = text.find("arith.constant").unwrap();
    let add = text.find("arith.addi").unwrap();
    let tag = text.find("analysis.tag = \"hot\"").unwrap();
    assert!(constant < add && add < tag, "{text}");
}

#[test]
fn remove_attr_keeps_selection_comments_and_other_attributes() {
    let input = "module {\n  // Keep this comment.\n  \"test.a\"() {analysis.tag = \"hot\", other = \"keep\"} : () -> () // trailing a\n  \"test.b\"() {analysis.tag = \"cold\"} : () -> () // trailing b\n}\n";
    let output = run_stdin(
        r#"filter(op("test.a")) | remove_attr("analysis.tag")"#,
        input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains("analysis.tag"), "{text}");
    assert!(text.contains("other = \"keep\""), "{text}");
    assert!(text.contains("// Keep this comment."), "{text}");
    let selected_line = text.lines().find(|line| line.contains("test.a")).unwrap();
    assert!(selected_line.ends_with("// trailing a"), "{text}");
    assert!(!text.contains("// trailing b"), "{text}");
    assert!(!text.contains("test.b"), "{text}");
}

#[test]
fn remove_attr_absent_is_a_noop_and_composes_in_pipeline_order() {
    let absent_input = "module {\n  // Keep the first.\n  \"test.a\"() : () -> () // trailing a\n  // Keep the second.\n  \"test.b\"() : () -> () // trailing b\n}\n";
    let absent = run_stdin(
        r#"filter(op("test.a") or op("test.b")) | remove_attr("missing.tag")"#,
        absent_input,
    );
    assert!(
        absent.status.success(),
        "{}",
        String::from_utf8_lossy(&absent.stderr)
    );
    let absent_text = String::from_utf8(absent.stdout).unwrap();
    let absent_lines = absent_text.lines().collect::<Vec<_>>();
    let a_line = absent_lines
        .iter()
        .find(|line| line.contains("test.a"))
        .unwrap();
    let b_line = absent_lines
        .iter()
        .find(|line| line.contains("test.b"))
        .unwrap();
    assert!(a_line.ends_with("// trailing a"), "{absent_text}");
    assert!(b_line.ends_with("// trailing b"), "{absent_text}");
    let first_comment = absent_text.find("// Keep the first.").unwrap();
    let test_a = absent_text.find("test.a").unwrap();
    let second_comment = absent_text.find("// Keep the second.").unwrap();
    let test_b = absent_text.find("test.b").unwrap();
    assert!(first_comment < test_a && test_a < second_comment && second_comment < test_b);

    let mixed_input = "module {\n  // Comment for a.\n  \"test.a\"() {analysis.tag = \"hot\"} : () -> () // trailing a\n  // Comment for b.\n  \"test.b\"() : () -> () // trailing b\n}\n";
    let mixed = run_stdin(
        r#"filter(op("test.a") or op("test.b")) | remove_attr("analysis.tag")"#,
        mixed_input,
    );
    assert!(
        mixed.status.success(),
        "{}",
        String::from_utf8_lossy(&mixed.stderr)
    );
    let mixed_text = String::from_utf8(mixed.stdout).unwrap();
    assert!(!mixed_text.contains("analysis.tag"), "{mixed_text}");
    let comment_for_a = mixed_text.find("// Comment for a.").unwrap();
    let test_a = mixed_text.find("\"test.a\"").unwrap();
    let comment_for_b = mixed_text.find("// Comment for b.").unwrap();
    let test_b = mixed_text.find("\"test.b\"").unwrap();
    assert!(
        comment_for_a < test_a && test_a < comment_for_b && comment_for_b < test_b,
        "{mixed_text}"
    );
    assert_eq!(mixed_text.matches("// Comment for b.").count(), 1);
    let mixed_lines = mixed_text.lines().collect::<Vec<_>>();
    let a_line = mixed_lines
        .iter()
        .find(|line| line.contains("test.a"))
        .unwrap();
    let b_line = mixed_lines
        .iter()
        .find(|line| line.contains("test.b"))
        .unwrap();
    assert!(a_line.ends_with("// trailing a"), "{mixed_text}");
    assert!(b_line.ends_with("// trailing b"), "{mixed_text}");

    let output = run_stdin(
        r#"filter(op("arith.addi")) | set_attr("analysis.tag", "hot") | remove_attr("analysis.tag") | fixpoint(closure) | input | emit"#,
        INPUT,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains("analysis.tag"), "{text}");
    assert!(text.contains("arith.constant"), "{text}");
    assert!(text.contains("example.observe"), "{text}");
}

#[test]
fn remove_attr_rejects_unparsed_operations_before_editing_the_selection() {
    for input in [
        "module {\n  mystery.consume {debug.note = \"sealed\"}\n}\n",
        "module {\n  \"example.known\"() {debug.note = \"drop\"} : () -> ()\n  mystery.consume {debug.note = \"sealed\"}\n}\n",
    ] {
        let output = run_stdin(
            r#"filter(op("example.known") or op("mystery.consume")) | remove_attr("debug.note")"#,
            input,
        );
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("cannot edit an incomplete document"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn remove_attr_does_not_attach_a_parent_tail_to_a_nested_operation() {
    for child_attributes in [" {analysis.tag = \"hot\"}", ""] {
        let input = format!(
            "module {{\n  \"test.parent\"() ({{ \"test.child\"(){child_attributes} : () -> () }}) : () -> () // parent tail\n}}\n"
        );
        let output = run_stdin(
            r#"filter(op("test.parent") or op("test.child")) | remove_attr("analysis.tag")"#,
            &input,
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert_eq!(text.matches("// parent tail").count(), 1, "{text}");
        let child_line = text
            .lines()
            .find(|line| line.contains("test.child"))
            .unwrap();
        assert!(!child_line.contains("// parent tail"), "{text}");
        let parent_tail = text
            .lines()
            .find(|line| line.contains("// parent tail"))
            .unwrap();
        assert!(parent_tail.trim_start().starts_with('}'), "{text}");
    }
}

#[test]
fn rejected_composition_produces_no_output() {
    let output = run_stdin("filter(op(\"arith.addi\")) | count | input | emit", INPUT);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no stage may follow count"));
}

#[test]
fn invalid_set_attr_text_produces_no_output() {
    for query in [
        "filter(op(\"arith.addi\")) | set_attr(\"bad name\", \"hot\") | input | emit",
        "filter(op(\"arith.addi\")) | set_attr(\"tag\", \"hot\nvalue\") | input | emit",
    ] {
        let output = run_stdin(query, INPUT);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("query error"));
    }
}

#[test]
fn escaped_set_attr_whole_document_output_is_parseable() {
    let input = "module {\n  %c = arith.constant 7 : i32\n  %sum = arith.addi %c, %c : i32\n}\n";
    let output = run_stdin(
        r#"filter(op("arith.addi")) | set_attr("analysis.tag", "say \"hi\" \\ path") | input | emit"#,
        input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(r#"analysis.tag = "say \"hi\" \\ path""#));
    let reparsed = run_stdin("filter(op(\"builtin.module\"))", &text);
    assert!(
        reparsed.status.success(),
        "{}",
        String::from_utf8_lossy(&reparsed.stderr)
    );
}

#[test]
fn closure_at_function_argument_retains_complete_function_scope() {
    let input = "\"builtin.module\"() ({\n  \"func.func\"() ({\n  ^entry(%arg0: i32):\n    %sum = \"arith.addi\"(%arg0, %arg0) : (i32, i32) -> i32\n    \"func.return\"(%sum) : (i32) -> ()\n  }) : () -> ()\n  \"example.other\"() : () -> ()\n}) : () -> ()\n";
    let output = run_stdin("filter(op(\"arith.addi\")) | fixpoint(closure)", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("\"func.func\""), "{text}");
    assert!(text.contains("func.return %"), "{text}");
    assert!(!text.contains("example.other"), "{text}");
}

#[test]
fn closure_follows_recursive_symbol_once_without_sibling_symbols() {
    let input = "module {\n  func.func @recursive() {\n    func.call @recursive() : () -> ()\n    func.return\n  }\n  func.func @unrelated() { func.return }\n  func.func @caller() {\n    func.call @recursive() : () -> ()\n    func.return\n  }\n}\n";
    let output = run_stdin("filter(op(\"func.call\")) | fixpoint(closure)", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("func.func @recursive").count(), 1, "{text}");
    assert_eq!(text.matches("func.call @recursive").count(), 2, "{text}");
    assert!(!text.contains("func.func @unrelated"), "{text}");
}

#[test]
fn closure_resolves_a_quoted_symbol_containing_path_separators() {
    let input = r#"module {
  func.func @"a::b"() { func.return }
  func.func @unrelated() { func.return }
  func.func @caller() {
    func.call @"a::b"() : () -> ()
    func.return
  }
}
"#;
    let output = run_stdin("filter(op(\"func.call\")) | fixpoint(closure)", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(r#"func.func @"a::b""#), "{text}");
    assert!(text.contains(r#"func.call @"a::b""#), "{text}");
    assert!(!text.contains("func.func @unrelated"), "{text}");
}

#[test]
fn closure_retains_cyclic_cfg_once_for_conditional_and_unconditional_branches() {
    let input = "module {\n  func.func @loop() {\n  ^entry:\n    cf.br ^loop\n  ^loop:\n    %condition = arith.constant 1 : i1\n    cf.cond_br %condition, ^loop, ^exit\n  ^exit:\n    func.return\n  }\n}\n";
    let output = run_stdin("filter(op(\"cf.cond_br\")) | fixpoint(closure)", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("func.func @loop").count(), 1, "{text}");
    assert_eq!(text.matches("cf.br").count(), 1, "{text}");
    assert_eq!(text.matches("cf.cond_br").count(), 1, "{text}");
    assert_eq!(text.matches("arith.constant").count(), 1, "{text}");
    assert_eq!(text.matches("func.return").count(), 1, "{text}");
}

#[test]
fn closure_reports_unresolved_callee_from_strict_lowering() {
    let input = "module {\n  func.func @caller() {\n    func.call @missing() : () -> ()\n    func.return\n  }\n}\n";
    let output = run_stdin("filter(op(\"func.call\")) | fixpoint(closure)", input);
    assert!(!output.status.success());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostic.contains("could not resolve func.call callee `@missing`"),
        "{diagnostic}"
    );
}

#[test]
fn closure_reports_invalid_successor_from_strict_lowering() {
    let input = "module {\n  func.func @caller() {\n    cf.br ^missing\n  }\n}\n";
    let output = run_stdin("filter(op(\"cf.br\")) | fixpoint(closure)", input);
    assert!(!output.status.success());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostic.contains("unresolved block `^missing`"),
        "{diagnostic}"
    );
}

#[test]
fn closure_rejects_symbol_reference_on_unregistered_operation() {
    let input = "\"example.def\"() {sym_name = \"callee\"} : () -> ()\n\"example.call\"() {callee = @callee} : () -> ()\n";
    let output = run_stdin("filter(op(\"example.call\")) | fixpoint(closure)", input);
    assert!(!output.status.success());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostic.contains(
            "cannot determine reference semantics for unregistered operation `example.call`"
        ),
        "{diagnostic}"
    );
}

#[test]
fn short_program_file_flag_reads_query_with_final_newline() {
    let program = temporary_path("short-program", "zirium");
    fs::write(&program, "filter(op(\"arith.addi\"))\n").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("-f")
        .arg(&program)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(INPUT.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let _ = fs::remove_file(program);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("arith.addi")
    );
}

#[test]
fn long_program_file_flag_keeps_mlir_file_arguments() {
    let program = temporary_path("long-program", "zirium");
    let first = temporary_path("long-first", "mlir");
    let second = temporary_path("long-second", "mlir");
    fs::write(&program, "filter(op(\"missing\"))").unwrap();
    fs::write(&first, INPUT).unwrap();
    fs::write(&second, INPUT).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("--program-file")
        .arg(&program)
        .arg(&first)
        .arg(&second)
        .output()
        .unwrap();
    let _ = fs::remove_file(program);
    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "// -----\n");
}

#[test]
fn unreadable_program_file_fails_usefully() {
    let missing = temporary_path("missing-program", "zirium");
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("-f")
        .arg(&missing)
        .output()
        .unwrap();
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        diagnostic.contains("could not read program file"),
        "{diagnostic}"
    );
}

#[test]
fn selected_root_retains_owned_contents_and_comments() {
    let input = "// Root comment.\nmodule {\n  // Function comment.\n  func.func @f() {\n    // Nested comment.\n    func.return\n  }\n}\n";
    let output = run_stdin("filter(op(\"builtin.module\"))", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        text,
        "// Root comment.\nbuiltin.module {\n  // Function comment.\n  func.func @f() -> () {\n    // Nested comment.\n    func.return\n  }\n}\n"
    );
}

#[test]
fn selected_function_with_argument_can_be_queried_again() {
    let input = "module { func.func @identity(%arg: i32) -> i32 { func.return %arg : i32 } }\n";
    let selected = run_stdin(r#"filter(op("func.func"))"#, input);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let printed = String::from_utf8(selected.stdout).unwrap();
    let reparsed = run_stdin(r#"filter(op("func.func")) | count"#, &printed);
    assert!(
        reparsed.status.success(),
        "{printed}\n{}",
        String::from_utf8_lossy(&reparsed.stderr)
    );
    assert_eq!(reparsed.stdout, b"1\n", "{printed}");
}

#[test]
fn selected_unknown_operation_preserves_enclosing_argument_names() {
    let input = "module {\n  func.func @pipeline(%source: i64) {\n    mystery.observe %source : i64\n    func.return\n  }\n}\n";
    let selected = run_stdin(r#"filter(op("mystery.observe"))"#, input);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let printed = String::from_utf8(selected.stdout).unwrap();
    assert!(
        printed.contains("func.func @pipeline(%source: i64)"),
        "{printed}"
    );
    assert!(
        printed.contains("mystery.observe %source : i64"),
        "{printed}"
    );

    for (query, expected) in [
        (
            r#"filter(op("mystery.observe")) | count"#,
            b"1\n".as_slice(),
        ),
        (
            r#"filter(op("mystery.observe")) | parent | count"#,
            b"1\n".as_slice(),
        ),
        (
            r#"filter(op("func.func")) | children | count"#,
            b"1\n".as_slice(),
        ),
    ] {
        let reparsed = run_stdin(query, &printed);
        assert!(
            reparsed.status.success(),
            "{printed}\n{}",
            String::from_utf8_lossy(&reparsed.stderr)
        );
        assert_eq!(reparsed.stdout, expected, "{printed}");
    }
}

#[test]
fn selected_unknown_operation_does_not_rewrite_attribute_strings() {
    let input = "module {\n  func.func @f(%arg: i32) attributes {note = \"%v0\"} {\n    vendor.use %arg : i32\n  }\n}\n";
    let selected = run_stdin(r#"filter(op("vendor.use"))"#, input);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let printed = String::from_utf8(selected.stdout).unwrap();
    assert!(printed.contains("func.func @f(%arg: i32)"), "{printed}");
    assert!(printed.contains("note = \"%v0\""), "{printed}");
    assert!(printed.contains("vendor.use %arg : i32"), "{printed}");
}

#[test]
fn inline_nested_operation_does_not_steal_parent_comment() {
    let input = "module {\n  // Nested function.\n  func.func @inner() { func.return }\n}\n";
    let output = run_stdin("filter(op(\"builtin.module\"))", input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("// Nested function.").count(), 1, "{text}");
    assert!(text.contains("func.func @inner() -> ()"), "{text}");
    assert!(text.contains("func.return"), "{text}");
}

#[test]
fn files_frame_empty_answers() {
    let directory = std::env::temp_dir();
    let first = directory.join(format!("zirium-cli-{}-a.mlir", std::process::id()));
    let second = directory.join(format!("zirium-cli-{}-b.mlir", std::process::id()));
    fs::write(&first, INPUT).unwrap();
    fs::write(&second, INPUT).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("filter(op(\"missing\"))")
        .arg(&first)
        .arg(&second)
        .output()
        .unwrap();
    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "// -----\n");
}

#[test]
fn malformed_query_and_input_fail_usefully() {
    let bad_query = run_stdin("filter(op(\"arith.addi\")", INPUT);
    assert!(!bad_query.status.success());
    assert!(String::from_utf8_lossy(&bad_query.stderr).contains("query error at byte"));
    let bad_input = run_stdin(
        "filter(op(\"arith.addi\"))",
        "module {\n  %x = arith.constant nope : i32\n}\n",
    );
    assert!(!bad_input.status.success());
    let diagnostic = String::from_utf8_lossy(&bad_input.stderr);
    assert!(diagnostic.contains("could not parse stdin"), "{diagnostic}");
    assert!(diagnostic.contains("Syntax at bytes"), "{diagnostic}");
}

#[test]
fn lowering_failure_reports_identity_and_original_range() {
    let input = "module {\n  \"example.use\"(%missing) : (i32) -> ()\n}\n";
    let output = run_stdin("filter(op(\"example.use\"))", input);
    assert!(!output.status.success());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostic.contains("diagnostic #1 at bytes"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("unresolved SSA value `%missing`"),
        "{diagnostic}"
    );
}

#[test]
fn arbitrary_generic_dialect_operations_support_structural_queries_and_edits() {
    let input = "\"builtin.module\"() ({\n  \"vendor.container\"() ({\n    \"vendor.compute\"() {tag = \"hot\", remove = \"yes\"} : () -> ()\n  }) : () -> ()\n}) : () -> ()\n";

    for (query, expected) in [
        (r#"filter(string_attr_eq("tag", "hot")) | count"#, "1\n"),
        (r#"filter(op("vendor.compute")) | parent | count"#, "1\n"),
        (
            r#"filter(op("vendor.container")) | children | count"#,
            "1\n",
        ),
    ] {
        let output = run_stdin(query, input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    }

    let edited = run_stdin(
        r#"filter(op("vendor.compute")) | set_attr("added", "value") | remove_attr("remove")"#,
        input,
    );
    assert!(
        edited.status.success(),
        "{}",
        String::from_utf8_lossy(&edited.stderr)
    );
    let text = String::from_utf8(edited.stdout).unwrap();
    assert!(text.contains("added = \"value\""), "{text}");
    assert!(!text.contains("remove ="), "{text}");
}

#[test]
fn bounded_unknown_custom_operation_supports_name_count_ownership_and_exact_selection() {
    let input = "module {\n  vendor.compute strangely<balanced>(payload)\n}\n";
    let count = run_stdin(r#"filter(op("vendor.compute")) | count"#, input);
    assert!(
        count.status.success(),
        "{}",
        String::from_utf8_lossy(&count.stderr)
    );
    assert_eq!(count.stdout, b"1\n");

    let parent_count = run_stdin(r#"filter(op("vendor.compute")) | parent | count"#, input);
    assert!(parent_count.status.success());
    assert_eq!(parent_count.stdout, b"1\n");

    let child_count = run_stdin(r#"filter(op("builtin.module")) | children | count"#, input);
    assert!(child_count.status.success());
    assert_eq!(child_count.stdout, b"1\n");

    let selected = run_stdin(r#"filter(op("vendor.compute"))"#, input);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    assert_eq!(
        String::from_utf8(selected.stdout).unwrap(),
        "builtin.module {\n  vendor.compute strangely<balanced>(payload)\n\n}\n"
    );
}

#[test]
fn unknown_custom_recovery_rejects_malformed_neighbors_closure_and_edits() {
    let malformed = run_stdin(
        r#"filter(op("vendor.compute"))"#,
        "module { vendor.compute strangely<unclosed(payload) }\n",
    );
    assert!(!malformed.status.success());
    assert!(malformed.stdout.is_empty());

    let input = "module { vendor.compute strangely<balanced>(payload) }\n";
    let closure = run_stdin(r#"filter(op("vendor.compute")) | fixpoint(closure)"#, input);
    assert!(!closure.status.success());
    assert!(closure.stdout.is_empty());
    assert!(String::from_utf8_lossy(&closure.stderr).contains(
        "cannot determine reference semantics for unregistered operation `vendor.compute`"
    ));

    let edit = run_stdin(
        r#"filter(op("vendor.compute")) | set_attr("tag", "value")"#,
        input,
    );
    assert!(!edit.status.success());
    assert!(edit.stdout.is_empty());
    assert!(String::from_utf8_lossy(&edit.stderr).contains("edit failed"));
}

#[test]
fn mutation_before_unknown_closure_failure_emits_no_stdout() {
    let input = "module {\n  \"vendor.known\"() : () -> ()\n  vendor.unknown opaque<payload>\n}\n";
    let output = run_stdin(
        r#"filter(op("vendor.known")) | set_attr("tag", "value") | (filter(true) union (input | filter(op("vendor.unknown")))) | fixpoint(closure)"#,
        input,
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn recovered_unknown_sibling_does_not_block_empty_or_understood_selections() {
    let input = "module {\n  \"vendor.known\"() : () -> ()\n  vendor.unknown opaque<payload>\n}\n";

    let empty = run_stdin(r#"filter(op("vendor.missing"))"#, input);
    assert!(
        empty.status.success(),
        "{}",
        String::from_utf8_lossy(&empty.stderr)
    );
    assert!(empty.stdout.is_empty());

    let understood = run_stdin(r#"filter(op("vendor.known"))"#, input);
    assert!(
        understood.status.success(),
        "{}",
        String::from_utf8_lossy(&understood.stderr)
    );
    let text = String::from_utf8(understood.stdout).unwrap();
    assert!(text.contains("vendor.known"), "{text}");
    assert!(!text.contains("vendor.unknown"), "{text}");
}

#[test]
fn named_nested_and_commented_module_shorthand_uses_the_parser() {
    let source = "module @outer attributes {tag = \"keep\"} {\n module // nested module\n @inner {\n  \"test.op\"() : () -> ()\n }\n}\n";
    let output = run_stdin(r#"filter(op("builtin.module")) | count"#, source);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"2\n");
    let printed = run_stdin(r#"filter(op("builtin.module")) | input | emit"#, source);
    assert!(
        printed.status.success(),
        "{}",
        String::from_utf8_lossy(&printed.stderr)
    );
    let restored = run_stdin(
        r#"filter(op("test.op")) | count"#,
        std::str::from_utf8(&printed.stdout).unwrap(),
    );
    assert!(
        restored.status.success(),
        "{}",
        String::from_utf8_lossy(&restored.stderr)
    );
    assert_eq!(restored.stdout, b"1\n");
}

#[test]
fn unknown_custom_sibling_does_not_hide_semantic_errors() {
    let source =
        "module {\n  \"test.use\"(%missing) : (i32) -> ()\n  vendor.unknown opaque<payload>\n}\n";
    let output = run_stdin(r#"filter(op("test.use")) | count"#, source);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostic.contains("could not lower stdin"), "{diagnostic}");
    assert!(diagnostic.contains("missing"), "{diagnostic}");
}

#[test]
fn registry_file_drives_cli_lowering_and_round_trip_output() {
    use zirium::{
        dialect::DialectRegistry,
        parser::ParsedFile,
        semantic::{LoweringMode, lower_with_dialect_registry},
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/cli");
    let registry_path = root.join("registry.json");
    let input = root.join("registered-shapes.mlir");
    let registry = DialectRegistry::from_config_file(&registry_path).unwrap();
    let source = fs::read(&input).unwrap();
    let parsed = ParsedFile::parse_with_registry(source, &registry).unwrap();
    let original = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry)
        .document
        .unwrap();
    for (query, expected) in [
        (
            r#"filter(op("vendor.function")) | count"#,
            Some(b"1\n".as_slice()),
        ),
        (r#"filter(op("vendor.invoke")) | input | emit"#, None),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
            .arg("--registry")
            .arg(&registry_path)
            .arg("--registry")
            .arg(&registry_path)
            .arg(query)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(expected) = expected {
            assert_eq!(output.stdout, expected);
        } else {
            let parsed = ParsedFile::parse_with_registry(output.stdout, &registry).unwrap();
            assert!(parsed.syntax().diagnostics().is_empty());
            let restored = lower_with_dialect_registry(&parsed, LoweringMode::Strict, &registry)
                .document
                .unwrap();
            assert!(original.structurally_eq(&restored));
        }
    }
    let program = temporary_path("registered-query", "zirium");
    fs::write(&program, r#"filter(op("vendor.invoke")) | count"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("-f")
        .arg(&program)
        .arg("--registry")
        .arg(&registry_path)
        .arg(&input)
        .output()
        .unwrap();
    fs::remove_file(program).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"1\n");
    let closure = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("--registry")
        .arg(&registry_path)
        .arg(r#"filter(op("vendor.invoke")) | fixpoint(closure)"#)
        .arg(&input)
        .output()
        .unwrap();
    assert!(!closure.status.success());
    assert!(closure.stdout.is_empty());
    assert!(String::from_utf8_lossy(&closure.stderr).contains("reference semantics"));
}

#[test]
fn invalid_registry_fails_without_waiting_for_mlir_stdin() {
    let registry = temporary_path("invalid-registry", "json");
    fs::write(
        &registry,
        r#"{"builtins":[],"operation_shapes":[],"typo":true}"#,
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .arg("--registry")
        .arg(&registry)
        .arg(r#"filter(op("a.b")) | count"#)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("read stdin before rejecting registry");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    fs::remove_file(registry).unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
}

#[test]
fn implicit_input_and_emit_cover_empty_programs_and_explicit_taps() {
    let expected = run_stdin("input | emit", INPUT);
    assert!(expected.status.success());
    for query in ["", " \n # unchanged input\n", "input", "emit", "(emit)"] {
        let output = run_stdin(query, INPUT);
        assert!(
            output.status.success(),
            "{query}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected.stdout, "{query}");
    }
    let twice = run_stdin("emit | emit", INPUT);
    assert!(twice.status.success());
    let parts = String::from_utf8(twice.stdout).unwrap();
    assert_eq!(
        parts.split("// -----\n").collect::<Vec<_>>(),
        vec![String::from_utf8_lossy(&expected.stdout); 2]
    );
}

#[test]
fn root_expands_the_selected_fragment_without_selecting_siblings() {
    let input = include_str!("../../../examples/cli/calls.mlir");
    for (query, expected) in [
        (r#"filter(op("func.call")) | parent | count"#, "1\n"),
        (r#"filter(op("func.call")) | parent | root | count"#, "3\n"),
        (
            r#"filter(op("func.call")) | parent | root | filter(op("func.return")) | count"#,
            "1\n",
        ),
        (
            r#"filter(op("func.call")) | parent | root | input | filter(op("func.return")) | count"#,
            "3\n",
        ),
        (r#"filter(op("func.call")) | root | count"#, "1\n"),
        ("filter(false) | root | count", "0\n"),
        ("root | count", "9\n"),
    ] {
        let output = run_stdin(query, input);
        assert!(
            output.status.success(),
            "{query}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            expected,
            "{query}"
        );
    }
    let edited = run_stdin(
        r#"filter(op("func.call")) | parent | root | filter(op("func.return")) | set_attr("tag", "chosen") | input"#,
        input,
    );
    assert!(
        edited.status.success(),
        "{}",
        String::from_utf8_lossy(&edited.stderr)
    );
    assert_eq!(
        String::from_utf8(edited.stdout)
            .unwrap()
            .matches("tag = \"chosen\"")
            .count(),
        1
    );
}

#[test]
fn closure_is_one_step_and_fixpoint_repeats_arbitrary_selection_queries() {
    let input =
        include_str!("../../../examples/cli/arithmetic.mlir").replace("arith.muli", "arith.addi");
    for (query, expected) in [
        (
            r#"filter(has_attr("analysis.tag")) | closure | count"#,
            "3\n",
        ),
        (
            r#"filter(has_attr("analysis.tag")) | fixpoint(closure) | count"#,
            "4\n",
        ),
        (
            r#"filter(has_attr("analysis.tag")) | fixpoint(defs) | count"#,
            "0\n",
        ),
        (
            r#"filter(op("arith.addi") and not has_attr("analysis.tag")) | fixpoint(filter(true) union users) | count"#,
            "2\n",
        ),
        (
            r#"filter(op("arith.addi") and not has_attr("analysis.tag")) | (defs union users) | count"#,
            "3\n",
        ),
        (
            r#"filter(op("arith.addi") and not has_attr("analysis.tag")) | (defs intersect users) | count"#,
            "0\n",
        ),
        (
            r#"filter(has_attr("analysis.tag")) | (closure except defs) | count"#,
            "1\n",
        ),
        ("filter(false) | fixpoint(closure) | count", "0\n"),
        ("fixpoint(fixpoint(filter(true))) | count", "5\n"),
    ] {
        let output = run_stdin(query, &input);
        assert!(
            output.status.success(),
            "{query}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            expected,
            "{query}"
        );
    }
    // A complement alternates between the empty and complete selection.
    let cycle = run_stdin("fixpoint(input except filter(true))", &input);
    assert!(!cycle.status.success());
    assert!(cycle.stdout.is_empty());
    assert!(String::from_utf8_lossy(&cycle.stderr).contains("cycles"));
}

#[test]
fn emit_captures_each_pipeline_position_before_later_edits() {
    let output = run_stdin(
        r#"filter(op("arith.addi")) | emit | set_attr("tag", "new") | emit | remove_attr("tag")"#,
        INPUT,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let fragments = text.split("// -----\n").collect::<Vec<_>>();
    assert_eq!(fragments.len(), 3);
    assert!(!fragments[0].contains("tag ="));
    assert!(fragments[1].contains("tag = \"new\""));
    assert_eq!(fragments[0], fragments[2]);

    let output = run_stdin(r#"filter(op("arith.addi")) | emit | users"#, INPUT);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let fragments = text.split("// -----\n").collect::<Vec<_>>();
    assert_eq!(fragments.len(), 2);
    assert!(fragments[0].contains("arith.addi"));
    assert!(!fragments[0].contains("example.observe"));
    assert!(fragments[1].contains("example.observe"));

    let iterations = run_stdin(
        r#"filter(op("arith.addi")) | fixpoint(closure | emit) | count"#,
        INPUT,
    );
    assert!(iterations.status.success());
    let text = String::from_utf8(iterations.stdout).unwrap();
    assert_eq!(text.matches("arith.addi").count(), 2);
    assert_eq!(text.matches("arith.constant").count(), 2);
    assert!(text.ends_with("2\n"));

    let branches = run_stdin(
        r#"filter(op("arith.addi")) | (defs | emit union users | emit) | count"#,
        INPUT,
    );
    assert!(branches.status.success());
    let text = String::from_utf8(branches.stdout).unwrap();
    assert!(text.find("arith.constant").unwrap() < text.find("example.observe").unwrap());
    assert!(text.ends_with("2\n"));

    let failed = run_stdin(r#"emit | filter(op("example.observe")) | closure"#, INPUT);
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
}

#[test]
fn filters_and_input_observe_prior_edits() {
    let output = run_stdin(
        r#"filter(op("arith.addi")) | set_attr("tag", "new") | users | input | filter(string_attr_eq("tag", "new")) | count"#,
        INPUT,
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"1\n");
    let scoped = run_stdin(
        r#"filter(op("arith.addi")) | filter(op("arith.constant")) | count"#,
        INPUT,
    );
    assert!(scoped.status.success());
    assert_eq!(scoped.stdout, b"0\n");
}

#[test]
fn query_diagnostics_keep_program_file_lines_and_unicode_columns() {
    let program = temporary_path("query-diagnostics", "zirium");
    fs::write(&program, "\n# a comment\nfilter(op(\"é\")) | typo\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zirium"))
        .args(["-f", program.to_str().unwrap()])
        .output()
        .unwrap();
    let _ = fs::remove_file(program);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("line 3, column 19"), "{error}");
    assert!(error.contains("\n                  ^"), "{error}");
}

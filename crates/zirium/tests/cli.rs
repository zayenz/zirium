use std::{
    fs,
    io::Write,
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
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
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
    let output = run_stdin("select(op(\"arith.addi\"))", INPUT);
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
        r#"select((op("test.a") or op("test.b")) and has_attr("tag") and not attr("tag", "other"))"#,
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

    let decoded = run_stdin(r#"select(attr("tag", "say \"hi\""))"#, input);
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
fn malformed_predicates_produce_no_output() {
    for query in [
        r#"select(attr("tag" "value"))"#,
        r#"select(op("arith.addi") and)"#,
        r#"select(has_attr("bad name"))"#,
    ] {
        let output = run_stdin(query, INPUT);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("query error"));
    }
}

#[test]
fn closure_adds_shared_ssa_definition_once_with_comments() {
    let output = run_stdin("select(op(\"arith.addi\")) | closure", INPUT);
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
    let defs = run_stdin("select(op(\"arith.addi\")) | defs", INPUT);
    assert!(
        defs.status.success(),
        "{}",
        String::from_utf8_lossy(&defs.stderr)
    );
    let text = String::from_utf8(defs.stdout).unwrap();
    assert_eq!(text.matches("arith.constant").count(), 1, "{text}");
    assert!(!text.contains("arith.addi"), "{text}");

    let users = run_stdin("select(op(\"arith.constant\")) | users | count", INPUT);
    assert!(
        users.status.success(),
        "{}",
        String::from_utf8_lossy(&users.stderr)
    );
    assert_eq!(String::from_utf8(users.stdout).unwrap(), "1\n");

    let unused = run_stdin("select(op(\"example.observe\")) | users | count", INPUT);
    assert!(unused.status.success());
    assert_eq!(String::from_utf8(unused.stdout).unwrap(), "0\n");
}

#[test]
fn predicate_set_operations_are_ordered_and_composable() {
    let input = "module {\n  \"test.a\"() {group = \"keep\"} : () -> ()\n  \"test.b\"() {group = \"keep\"} : () -> ()\n  \"test.c\"() : () -> ()\n}\n";

    let union = run_stdin(
        r#"select(op("test.c")) | union(op("test.a") or has_attr("group"))"#,
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
        r#"select(has_attr("group")) | intersect(not op("test.b")) | count"#,
        input,
    );
    assert!(intersect.status.success());
    assert_eq!(String::from_utf8(intersect.stdout).unwrap(), "1\n");

    for query in [
        r#"select(has_attr("group")) | except(attr("group", "keep")) | count"#,
        r#"select(op("missing")) | union(op("missing")) | count"#,
        r#"select(op("missing")) | intersect(op("test.a")) | count"#,
        r#"select(op("test.a")) | except(op("missing")) | count"#,
    ] {
        let output = run_stdin(query, input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = if query.contains("except(op") {
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
        .arg(r#"select(op("test.a")) | union(op("test.d")) | count"#)
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

    let boundary = run_stdin("select(op(\"arith.addi\")) | defs", input);
    assert!(
        boundary.status.success(),
        "{}",
        String::from_utf8_lossy(&boundary.stderr)
    );
    let text = String::from_utf8(boundary.stdout).unwrap();
    assert!(text.contains("func.func"), "{text}");
    assert!(text.contains("arith.addi"), "{text}");

    let parent = run_stdin("select(op(\"arith.addi\")) | parent", input);
    assert!(parent.status.success());
    let text = String::from_utf8(parent.stdout).unwrap();
    assert!(text.contains("func.func"), "{text}");

    let children = run_stdin("select(op(\"func.func\")) | children", input);
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
        .arg("select(op(\"arith.addi\")) | count")
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
        "select(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\")",
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
fn root_after_mutation_prints_the_validated_whole_document() {
    let output = run_stdin(
        "select(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | root",
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
fn set_attr_mutates_the_selection_at_its_pipeline_position() {
    let output = run_stdin(
        "select(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | closure | root",
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
        r#"select(op("test.a")) | remove_attr("analysis.tag")"#,
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
        r#"select(op("test.a") or op("test.b")) | remove_attr("missing.tag")"#,
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
        r#"select(op("test.a") or op("test.b")) | remove_attr("analysis.tag")"#,
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
        r#"select(op("arith.addi")) | set_attr("analysis.tag", "hot") | remove_attr("analysis.tag") | closure | root"#,
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
fn remove_attr_does_not_attach_a_parent_tail_to_a_nested_operation() {
    for child_attributes in [" {analysis.tag = \"hot\"}", ""] {
        let input = format!(
            "module {{\n  \"test.parent\"() ({{ \"test.child\"(){child_attributes} : () -> () }}) : () -> () // parent tail\n}}\n"
        );
        let output = run_stdin(
            r#"select(op("test.parent") or op("test.child")) | remove_attr("analysis.tag")"#,
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
    let output = run_stdin("select(op(\"arith.addi\")) | count | root", INPUT);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("root requires a selection"));
}

#[test]
fn invalid_set_attr_text_produces_no_output() {
    for query in [
        "select(op(\"arith.addi\")) | set_attr(\"bad name\", \"hot\") | root",
        "select(op(\"arith.addi\")) | set_attr(\"tag\", \"hot\nvalue\") | root",
    ] {
        let output = run_stdin(query, INPUT);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("query error"));
    }
}

#[test]
fn escaped_set_attr_root_output_is_parseable() {
    let input = "module {\n  %c = arith.constant 7 : i32\n  %sum = arith.addi %c, %c : i32\n}\n";
    let output = run_stdin(
        r#"select(op("arith.addi")) | set_attr("analysis.tag", "say \"hi\" \\ path") | root"#,
        input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(r#"analysis.tag = "say \"hi\" \\ path""#));
    let reparsed = run_stdin("select(op(\"builtin.module\"))", &text);
    assert!(
        reparsed.status.success(),
        "{}",
        String::from_utf8_lossy(&reparsed.stderr)
    );
}

#[test]
fn closure_at_function_argument_retains_complete_function_scope() {
    let input = "\"builtin.module\"() ({\n  \"func.func\"() ({\n  ^entry(%arg0: i32):\n    %sum = \"arith.addi\"(%arg0, %arg0) : (i32, i32) -> i32\n    \"func.return\"(%sum) : (i32) -> ()\n  }) : () -> ()\n  \"example.other\"() : () -> ()\n}) : () -> ()\n";
    let output = run_stdin("select(op(\"arith.addi\")) | closure", input);
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
    let output = run_stdin("select(op(\"func.call\")) | closure", input);
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
fn closure_retains_cyclic_cfg_once_for_conditional_and_unconditional_branches() {
    let input = "module {\n  func.func @loop() {\n  ^entry:\n    cf.br ^loop\n  ^loop:\n    %condition = arith.constant 1 : i1\n    cf.cond_br %condition, ^loop, ^exit\n  ^exit:\n    func.return\n  }\n}\n";
    let output = run_stdin("select(op(\"cf.cond_br\")) | closure", input);
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
    let output = run_stdin("select(op(\"func.call\")) | closure", input);
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
    let output = run_stdin("select(op(\"cf.br\")) | closure", input);
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
    let output = run_stdin("select(op(\"example.call\")) | closure", input);
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
    fs::write(&program, "select(op(\"arith.addi\"))\n").unwrap();
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
    fs::write(&program, "select(op(\"missing\"))").unwrap();
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
    let output = run_stdin("select(op(\"builtin.module\"))", input);
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
fn inline_nested_operation_does_not_steal_parent_comment() {
    let input = "module {\n  // Nested function.\n  func.func @inner() { func.return }\n}\n";
    let output = run_stdin("select(op(\"builtin.module\"))", input);
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
        .arg("select(op(\"missing\"))")
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
    let bad_query = run_stdin("select(op(\"arith.addi\")", INPUT);
    assert!(!bad_query.status.success());
    assert!(String::from_utf8_lossy(&bad_query.stderr).contains("query error at byte"));
    let bad_input = run_stdin(
        "select(op(\"arith.addi\"))",
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
    let output = run_stdin("select(op(\"example.use\"))", input);
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
        (r#"select(attr("tag", "hot")) | count"#, "1\n"),
        (r#"select(op("vendor.compute")) | parent | count"#, "1\n"),
        (
            r#"select(op("vendor.container")) | children | count"#,
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
        r#"select(op("vendor.compute")) | set_attr("added", "value") | remove_attr("remove")"#,
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
    let count = run_stdin(r#"select(op("vendor.compute")) | count"#, input);
    assert!(
        count.status.success(),
        "{}",
        String::from_utf8_lossy(&count.stderr)
    );
    assert_eq!(count.stdout, b"1\n");

    let parent_count = run_stdin(r#"select(op("vendor.compute")) | parent | count"#, input);
    assert!(parent_count.status.success());
    assert_eq!(parent_count.stdout, b"1\n");

    let child_count = run_stdin(r#"select(op("builtin.module")) | children | count"#, input);
    assert!(child_count.status.success());
    assert_eq!(child_count.stdout, b"1\n");

    let selected = run_stdin(r#"select(op("vendor.compute"))"#, input);
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
        r#"select(op("vendor.compute"))"#,
        "module { vendor.compute strangely<unclosed(payload) }\n",
    );
    assert!(!malformed.status.success());
    assert!(malformed.stdout.is_empty());

    let input = "module { vendor.compute strangely<balanced>(payload) }\n";
    let closure = run_stdin(r#"select(op("vendor.compute")) | closure"#, input);
    assert!(!closure.status.success());
    assert!(closure.stdout.is_empty());
    assert!(String::from_utf8_lossy(&closure.stderr).contains(
        "cannot determine reference semantics for unregistered operation `vendor.compute`"
    ));

    let edit = run_stdin(
        r#"select(op("vendor.compute")) | set_attr("tag", "value")"#,
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
        r#"select(op("vendor.known")) | set_attr("tag", "value") | union(op("vendor.unknown")) | closure"#,
        input,
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn recovered_unknown_sibling_does_not_block_empty_or_understood_selections() {
    let input = "module {\n  \"vendor.known\"() : () -> ()\n  vendor.unknown opaque<payload>\n}\n";

    let empty = run_stdin(r#"select(op("vendor.missing"))"#, input);
    assert!(
        empty.status.success(),
        "{}",
        String::from_utf8_lossy(&empty.stderr)
    );
    assert!(empty.stdout.is_empty());

    let understood = run_stdin(r#"select(op("vendor.known"))"#, input);
    assert!(
        understood.status.success(),
        "{}",
        String::from_utf8_lossy(&understood.stderr)
    );
    let text = String::from_utf8(understood.stdout).unwrap();
    assert!(text.contains("vendor.known"), "{text}");
    assert!(!text.contains("vendor.unknown"), "{text}");
}

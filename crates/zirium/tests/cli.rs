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

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
    assert!(
        text.contains("attributes: value = 4 -> value = 8"),
        "{text}"
    );

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

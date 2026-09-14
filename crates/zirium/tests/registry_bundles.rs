use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use zirium::dialect::{
    DeclarativeRegistryError, DialectRegistry, OperationAlternative, RegistryConfig,
    RegistryConfigError, RegistryLoadOptions,
};
use zirium::{
    parser::ParsedFile,
    semantic::{LoweringMode, ValueId, ValueReference, lower_with_dialect_registry},
};

const WIDEN_SOURCE: &[u8] = br#""builtin.module"() ({
^bb0:
  %lhs = "test.source"() : () -> i16
  %rhs = "test.source"() : () -> i16
  %result = vendor.widen %lhs, %rhs {tag = true} : i16 to i32
}) : () -> ()"#;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/registry-bundles")
        .join(name)
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zirium-registry-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn empty(imports: &str) -> String {
    format!(r#"{{"imports":{imports},"builtins":[],"operation_shapes":[]}}"#)
}

fn assert_bundle_behavior(registry: &DialectRegistry) {
    assert_eq!(
        registry.call_target_attribute("vendor.invoke"),
        Some("target")
    );
    assert_eq!(
        registry.operation_alternatives("vendor.choice"),
        Some(vec![
            OperationAlternative::Format("$value `:` type($value) attr-dict `:` type($result)"),
            OperationAlternative::Shape(zirium::dialect::OperationShape::OperandClauses),
        ])
    );

    let parsed = ParsedFile::parse_with_registry(WIDEN_SOURCE, registry).unwrap();
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let lowered = lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry);
    assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);
    let document = lowered.document.unwrap();
    let operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("vendor.widen"))
        .unwrap();
    let operand_types = document
        .operands(operation)
        .unwrap()
        .iter()
        .map(|operand| match operand {
            ValueReference::Resolved(ValueId::OperationResult { operation, result }) => document
                .result_types(*operation)
                .and_then(|types| types.get(*result as usize))
                .and_then(|ty| document.type_spelling(*ty)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(operand_types, [Some("i16"), Some("i16")]);
    assert_eq!(
        document
            .result_types(operation)
            .unwrap()
            .iter()
            .map(|ty| document.type_spelling(*ty).unwrap())
            .collect::<Vec<_>>(),
        ["i32"]
    );
    assert!(
        document
            .attributes(operation)
            .unwrap()
            .any(|(name, value)| name == "tag" && value == "true")
    );
}

#[test]
fn three_leaf_bundle_matches_direct_composition_and_survives_a_move() {
    let bundled = DialectRegistry::from_config_file(fixture("root.json")).unwrap();
    let direct = DialectRegistry::from_config_files([
        fixture("leaves/builtins.json"),
        fixture("leaves/shapes.json"),
        fixture("leaves/formats.json"),
    ])
    .unwrap();
    assert_eq!(
        bundled.operation_names().collect::<Vec<_>>(),
        direct.operation_names().collect::<Vec<_>>()
    );
    assert_bundle_behavior(&bundled);
    assert_bundle_behavior(&direct);

    let moved = TempDir::new("moved");
    fs::create_dir(moved.0.join("leaves")).unwrap();
    for relative in [
        "root.json",
        "leaves/builtins.json",
        "leaves/shapes.json",
        "leaves/formats.json",
    ] {
        fs::copy(fixture(relative), moved.0.join(relative)).unwrap();
    }
    let moved_registry = DialectRegistry::from_config_file(moved.0.join("root.json")).unwrap();
    assert_bundle_behavior(&moved_registry);
}

#[test]
fn nested_diamonds_and_repeated_roots_deduplicate_across_parents() {
    let temp = TempDir::new("diamond");
    let leaf = temp.write(
        "leaf.json",
        r#"{"builtins":[],"operation_shapes":[{"name":"vendor.leaf","shape":"unary_operand"}]}"#,
    );
    temp.write("left.json", &empty(r#"["leaf.json"]"#));
    temp.write("right.json", &empty(r#"["leaf.json"]"#));
    let root = temp.write("root.json", &empty(r#"["left.json","right.json"]"#));
    let registry = DialectRegistry::from_config_files([&root, &root, &leaf]).unwrap();
    assert_eq!(
        registry.operation_shape("vendor.leaf").unwrap().name(),
        "unary_operand"
    );
}

#[test]
fn cycles_and_duplicate_canonical_siblings_are_rejected() {
    let temp = TempDir::new("invalid-graph");
    let root = temp.write("root.json", &empty(r#"["parent.json"]"#));
    temp.write("parent.json", &empty(r#"["a.json"]"#));
    temp.write("a.json", &empty(r#"["b.json"]"#));
    temp.write("b.json", &empty(r#"["a.json"]"#));
    let error = DialectRegistry::from_config_file(&root)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("cycle"), "{error}");
    let root_at = error.find("root.json").unwrap();
    let parent_at = error[root_at..].find("parent.json").unwrap() + root_at;
    let a_at = error[parent_at..].find("a.json").unwrap() + parent_at;
    let b_at = error[a_at..].find("b.json").unwrap() + a_at;
    let repeated_a_at = error[b_at..].find("a.json").unwrap() + b_at;
    assert!(root_at < parent_at && parent_at < a_at && a_at < b_at && b_at < repeated_a_at);
    assert!(error.contains("(cycle:"), "{error}");

    temp.write("child.json", &empty("[]"));
    let duplicate = temp.write("duplicate.json", &empty(r#"["child.json","./child.json"]"#));
    let error = DialectRegistry::from_config_file(duplicate)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("same canonical child"), "{error}");
    assert!(
        error.contains("child.json") && error.contains("./child.json"),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn symlink_aliases_are_duplicates_for_one_parent_and_deduplicated_across_parents() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new("symlink");
    temp.write("leaf.json", &empty("[]"));
    symlink("leaf.json", temp.0.join("alias.json")).unwrap();
    let duplicate = temp.write("duplicate.json", &empty(r#"["leaf.json","alias.json"]"#));
    assert!(
        DialectRegistry::from_config_file(duplicate)
            .err()
            .unwrap()
            .to_string()
            .contains("same canonical child")
    );

    temp.write("left.json", &empty(r#"["leaf.json"]"#));
    temp.write("right.json", &empty(r#"["alias.json"]"#));
    let root = temp.write("root.json", &empty(r#"["left.json","right.json"]"#));
    DialectRegistry::from_config_file(root).unwrap();
}

#[test]
fn every_graph_budget_accepts_its_boundary_and_rejects_one_less() {
    let root = fixture("root.json");
    let files = [
        fixture("root.json"),
        fixture("leaves/builtins.json"),
        fixture("leaves/shapes.json"),
        fixture("leaves/formats.json"),
    ];
    let bytes = files
        .iter()
        .map(|path| fs::read(path).unwrap().len())
        .sum::<usize>();
    let exact = RegistryLoadOptions {
        max_depth: 1,
        max_files: 4,
        max_edges: 3,
        max_bytes: bytes,
    };
    DialectRegistry::from_config_files_with_options([&root], exact).unwrap();
    for limited in [
        RegistryLoadOptions {
            max_depth: 0,
            ..exact
        },
        RegistryLoadOptions {
            max_files: 3,
            ..exact
        },
        RegistryLoadOptions {
            max_edges: 2,
            ..exact
        },
        RegistryLoadOptions {
            max_bytes: bytes - 1,
            ..exact
        },
    ] {
        assert!(matches!(
            DialectRegistry::from_config_files_with_options([&root], limited),
            Err(RegistryConfigError::Limit(_))
        ));
    }
}

#[test]
fn validation_and_conflicts_report_sources_and_in_memory_imports_do_not_read() {
    for json in [
        r#"{"imports":[""],"builtins":[],"operation_shapes":[]}"#,
        r#"{"imports":["/absolute.json"],"builtins":[],"operation_shapes":[]}"#,
    ] {
        assert!(RegistryConfig::from_json(json).is_err());
    }
    let unresolved = RegistryConfig::from_json(
        r#"{"imports":["missing.json"],"builtins":[],"operation_shapes":[]}"#,
    )
    .unwrap();
    assert!(matches!(
        unresolved.build(),
        Err(DeclarativeRegistryError::UnresolvedImports)
    ));

    let temp = TempDir::new("conflict");
    temp.write(
        "left.json",
        r#"{"builtins":[],"operation_shapes":[{"name":"vendor.op","shape":"unary_operand"}]}"#,
    );
    temp.write(
        "right.json",
        r#"{"builtins":[],"operation_shapes":[{"name":"vendor.op","shape":"binary_operands"}]}"#,
    );
    let root = temp.write("root.json", &empty(r#"["left.json","right.json"]"#));
    let error = DialectRegistry::from_config_file(root)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("left.json") && error.contains("right.json"),
        "{error}"
    );
    assert!(error.contains("import chain"), "{error}");
}

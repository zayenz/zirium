use zirium::{
    dialect::{DialectRegistry, OperationShape},
    parser::ParsedFile,
};

fn parse(source: &str) -> ParsedFile {
    let registry =
        DialectRegistry::with_operation_shapes(&[("test.regions", OperationShape::RegionClauses)])
            .unwrap();
    ParsedFile::parse_with_registry(source.as_bytes(), &registry).unwrap()
}

fn assert_regions_and_following_operation(source: &str, expected_regions: usize) {
    let parsed = parse(source);
    assert!(
        parsed.syntax().diagnostics().is_empty(),
        "{:?}",
        parsed.syntax().diagnostics()
    );
    let operations = parsed.syntax().file().operations().collect::<Vec<_>>();
    let region_operation = operations
        .iter()
        .find(|operation| {
            operation
                .mnemonic_range()
                .is_some_and(|range| parsed.source().slice(range).unwrap() == b"test.regions")
        })
        .unwrap();
    assert_eq!(region_operation.regions().count(), expected_regions);
    assert_eq!(
        operations
            .iter()
            .filter(|operation| {
                operation.mnemonic_range().is_some_and(|range| {
                    parsed.source().slice(range).unwrap() == br#""test.after""#
                })
            })
            .count(),
        1
    );
    parsed.syntax().tree().verify().unwrap();
}

#[test]
fn labeled_regions_continue_only_the_region_owning_operation() {
    assert_regions_and_following_operation(
        r#"test.regions mode = "indexed" {
  "test.yield"() : () -> ()
}
case 0 {
  "test.yield"() : () -> ()
}
case -1 {
  "test.yield"() : () -> ()
}
default {
  "test.yield"() : () -> ()
}
"test.after"() {note = "separate"} : () -> ()"#,
        4,
    );
}

#[test]
fn comma_separated_sibling_regions_continue_only_the_region_owning_operation() {
    assert_regions_and_following_operation(
        r#"test.regions {
  "test.yield"() : () -> ()
}, {
  "test.yield"() : () -> ()
}
"test.after"() : () -> ()"#,
        2,
    );
}

#[test]
fn existing_else_and_adjacent_region_forms_remain_supported() {
    for source in [
        r#"test.regions {} else {}
"test.after"() : () -> ()"#,
        r#"test.regions {} {}
"test.after"() : () -> ()"#,
    ] {
        assert_regions_and_following_operation(source, 2);
    }
}

#[test]
fn named_following_operation_with_balanced_attributes_stays_separate() {
    assert_regions_and_following_operation(
        r#"test.regions config = {mode = "ordinary"} {}
"test.after"() {metadata = {case = 0}} : () -> ()"#,
        1,
    );
}

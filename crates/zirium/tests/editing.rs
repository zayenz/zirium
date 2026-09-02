use std::sync::Arc;

use zirium::{
    dialect::{AttributeDescriptor, DialectRegistry, TypeDescriptor},
    parser::ParsedFile,
    printer::PrintLayout,
    semantic::{
        AttributeSpec, EditError, InsertionPoint, LoweringMode, OperationSpec, RetentionProfile,
        SemanticVerificationError, SharedRegistry, TypeSpec, TypeValue, ValueId,
        lower_proving_fixture, lower_proving_fixture_with_retention, lower_with_dialect_registry,
    },
};

fn reject_value(_: &str) -> Result<(), &'static str> {
    Err("edited value rejected")
}

static REJECTED_TYPES: [TypeDescriptor; 1] = [TypeDescriptor {
    name: "!edit.rejected",
    parse: None,
    lower: None,
    verify: Some(reject_value),
    print: None,
}];
static REJECTED_ATTRIBUTES: [AttributeDescriptor; 1] = [AttributeDescriptor {
    name: "#edit.rejected",
    parse: None,
    lower: None,
    verify: Some(reject_value),
    print: None,
}];
static REJECTING_VALUE_REGISTRY: DialectRegistry =
    DialectRegistry::new(&[], &REJECTED_TYPES, &REJECTED_ATTRIBUTES);

fn hybrid(source: &[u8], mode: LoweringMode) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse(Arc::<[u8]>::from(source)).unwrap();
    lower_proving_fixture_with_retention(&parsed, mode, RetentionProfile::Hybrid, &SharedRegistry)
        .document
        .unwrap()
}

fn registered(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse_with_registry(
        Arc::<[u8]>::from(source.as_bytes()),
        DialectRegistry::proving(),
    )
    .unwrap();
    lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::proving())
        .document
        .unwrap()
}

fn registered_best_effort(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse_with_registry(
        Arc::<[u8]>::from(source.as_bytes()),
        DialectRegistry::proving(),
    )
    .unwrap();
    lower_with_dialect_registry(
        &parsed,
        LoweringMode::BestEffort,
        DialectRegistry::proving(),
    )
    .document
    .unwrap()
}

fn generic(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse(source.as_bytes()).unwrap();
    lower_proving_fixture(&parsed, LoweringMode::Strict, &SharedRegistry)
        .document
        .unwrap()
}

fn generic_best_effort(source: &str) -> zirium::semantic::Document {
    let parsed = ParsedFile::parse(source.as_bytes()).unwrap();
    lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry)
        .document
        .unwrap()
}

fn i32_type() -> TypeSpec {
    TypeSpec {
        spelling: "i32".into(),
        value: TypeValue::Integer {
            width: 32,
            signedness: None,
        },
    }
}

fn empty_function_type() -> TypeSpec {
    TypeSpec {
        spelling: "() -> ()".into(),
        value: TypeValue::Function {
            inputs: vec![],
            results: vec![],
        },
    }
}

fn unknown_spec(name: &str) -> OperationSpec {
    OperationSpec {
        name: name.into(),
        operands: vec![],
        result_types: vec![],
        function_type: empty_function_type(),
        attributes: vec![],
        properties: vec![],
    }
}

fn validate(document: &zirium::semantic::Document) {
    document.validate_structure().unwrap();
    document
        .verify_semantics(DialectRegistry::proving())
        .unwrap();
}

#[test]
fn editor_commit_runs_registered_type_and_attribute_verifiers() {
    let mut type_document = generic("%0 = \"make\"() : () -> i32");
    let operation = type_document.root_operations()[0];
    let mut editor = type_document.edit(&REJECTING_VALUE_REGISTRY).unwrap();
    editor
        .replace_result_types(
            operation,
            &[TypeSpec {
                spelling: "!edit.rejected<x>".into(),
                value: TypeValue::Opaque(Arc::from(b"!edit.rejected<x>".as_slice())),
            }],
        )
        .unwrap();
    assert!(matches!(
        editor.commit(),
        Err(EditError::Semantic(SemanticVerificationError::Type { .. }))
    ));

    let mut attribute_document = generic("\"use\"() : () -> ()");
    let operation = attribute_document.root_operations()[0];
    let mut editor = attribute_document.edit(&REJECTING_VALUE_REGISTRY).unwrap();
    editor
        .set_attribute(
            operation,
            AttributeSpec {
                name: "tag".into(),
                spelling: "#edit.rejected<x>".into(),
                value: zirium::semantic::AttributeValue::Opaque(Arc::from(
                    b"#edit.rejected<x>".as_slice(),
                )),
            },
        )
        .unwrap();
    assert!(matches!(
        editor.commit(),
        Err(EditError::Semantic(
            SemanticVerificationError::Attribute { .. }
        ))
    ));
}

#[test]
fn preserving_output_copies_unchanged_arbitrary_source_bytes() {
    let source = b"\xff malformed\n\"known\"() : () -> ()\nweird.custom ???\n";
    let document = hybrid(source, LoweringMode::BestEffort);
    assert_eq!(
        document.preserving_bytes(PrintLayout::Pretty).unwrap(),
        source
    );
}

#[test]
fn preserving_output_replaces_only_a_dirty_operation() {
    let source = b"// before\n\"left\"() : () -> ()\n\n\"right\"() : () -> () // after\n";
    let mut document = hybrid(source, LoweringMode::Strict);
    let right = document.root_operations()[1];
    let mut editor = document.edit(&DialectRegistry::EMPTY).unwrap();
    editor
        .set_attribute(
            right,
            AttributeSpec {
                name: "tag".into(),
                spelling: "\"new\"".into(),
                value: zirium::semantic::AttributeValue::String("\"new\"".into()),
            },
        )
        .unwrap();
    editor.commit().unwrap();
    let output = document.preserving_bytes(PrintLayout::Compact).unwrap();
    assert!(output.starts_with(b"// before\n\"left\"() : () -> ()\n\n"));
    assert!(output.ends_with(b" // after\n"));
    assert!(String::from_utf8(output).unwrap().contains("tag = \"new\""));
}

#[test]
fn result_type_edits_widen_preservation_to_the_enclosing_block() {
    let source = b"\"outer\"() ({\n  %x = \"make\"() : () -> i32\n  \"use\"(%x) : (i32) -> ()\n}) : () -> ()";
    let mut document = hybrid(source, LoweringMode::Strict);
    let outer = document.root_operations()[0];
    let region = document.operation_regions(outer).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    let make = document.block_operations(block).unwrap()[0];
    let mut editor = document.edit(&DialectRegistry::EMPTY).unwrap();
    editor.replace_result_types(make, &[i32_type()]).unwrap();
    editor.commit().unwrap();
    let output = document.preserving_bytes(PrintLayout::Pretty).unwrap();
    let reparsed = ParsedFile::parse(output).unwrap();
    reparsed.syntax().tree().verify().unwrap();
    let relowered = lower_proving_fixture(&reparsed, LoweringMode::Strict, &SharedRegistry)
        .document
        .unwrap();
    assert!(document.structurally_eq(&relowered));
}

#[test]
fn preserving_preflight_rejects_unknown_custom_syntax_before_sink_writes() {
    let source = b"\"outer\"() ({\n  \"make\"() : () -> ()\n  vendor.unknown ???\n}) : () -> ()";
    let mut document = hybrid(source, LoweringMode::BestEffort);
    let outer = document.root_operations()[0];
    let region = document.operation_regions(outer).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[0];
    let make = document.block_operations(block).unwrap()[0];
    let mut editor = document.edit(&DialectRegistry::EMPTY).unwrap();
    editor.replace_result_types(make, &[]).unwrap();
    editor.commit().unwrap();
    let mut sink = Vec::new();
    let error = document
        .write_preserving(&mut sink, PrintLayout::Pretty)
        .unwrap_err();
    assert!(matches!(
        error,
        zirium::printer::PreserveError::UnknownCustomSyntax(_)
    ));
    assert!(sink.is_empty());
}

#[test]
fn preserving_stream_propagates_sink_failures() {
    struct FailingSink {
        remaining: usize,
    }
    impl std::io::Write for FailingSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Err(std::io::Error::other("expected sink failure"));
            }
            let written = bytes.len().min(self.remaining);
            self.remaining -= written;
            Ok(written)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let document = hybrid(
        b"\"left\"() : () -> ()\n\"right\"() : () -> ()\n",
        LoweringMode::Strict,
    );
    let error = document
        .write_preserving(&mut FailingSink { remaining: 7 }, PrintLayout::Compact)
        .unwrap_err();
    assert!(matches!(error, zirium::printer::PreserveError::Io(_)));
}

#[test]
fn rewire_all_uses_then_erase_preserves_other_ids_and_stales_erased_ids() {
    let mut document = registered(
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b : i32",
    );
    let ids = document.operations().collect::<Vec<_>>();
    let erased_result = document
        .operation(ids[0])
        .unwrap()
        .result(ids[0], 0)
        .unwrap();
    let replacement = document
        .operation(ids[1])
        .unwrap()
        .result(ids[1], 0)
        .unwrap();

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert!(matches!(editor.erase(ids[0]), Err(EditError::LiveUses(id)) if id == ids[0]));
    editor.rewire_operand(ids[2], 0, replacement).unwrap();
    editor.erase(ids[0]).unwrap();
    let reused = editor
        .insert(InsertionPoint::Root(2), unknown_spec("replacement.slot"))
        .unwrap();
    editor.commit().unwrap();

    validate(&document);
    assert!(document.operation(ids[0]).is_none());
    assert_ne!(reused, ids[0]);
    assert!(document.operation(reused).is_some());
    assert!(document.operation(ids[1]).is_some());
    assert!(
        matches!(erased_result, ValueId::OperationResult { operation, .. } if document.operation(operation).is_none())
    );
}

#[test]
fn failed_commit_and_dropped_transaction_leave_original_untouched() {
    let mut document = registered("%c = arith.constant 1 : i32");
    let constant = document.root_operations()[0];
    let revision = document.revision();
    let result = document
        .edit(DialectRegistry::proving())
        .unwrap()
        .remove_attribute(constant, "value");
    assert!(result.is_ok());
    // The preceding editor was dropped without committing.
    assert!(document.attribute_id(constant, "value").is_some());

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor.remove_attribute(constant, "value").unwrap();
    assert!(matches!(editor.commit(), Err(EditError::Semantic(_))));
    assert!(document.attribute_id(constant, "value").is_some());
    assert_eq!(document.revision(), revision);
    validate(&document);
}

#[test]
fn fixed_result_type_edit_preserves_result_identity() {
    let mut document = generic("%x = \"unknown.value\"() : () -> i32");
    let operation = document.root_operations()[0];
    let result = document
        .operation(operation)
        .unwrap()
        .result(operation, 0)
        .unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert!(matches!(
        editor.replace_result_types(operation, &[]),
        Err(EditError::ResultCountChange)
    ));
    editor
        .replace_result_types(operation, &[i32_type()])
        .unwrap();
    editor.commit().unwrap();
    validate(&document);
    assert_eq!(
        document.operation(operation).unwrap().result(operation, 0),
        Some(result)
    );
}

#[test]
fn insertion_updates_root_and_nested_block_relationships() {
    let mut roots = generic("\"existing\"() : () -> ()");
    let existing = roots.root_operations()[0];
    let mut editor = roots.edit(DialectRegistry::proving()).unwrap();
    let inserted = editor
        .insert(InsertionPoint::Root(0), unknown_spec("inserted"))
        .unwrap();
    editor.commit().unwrap();
    validate(&roots);
    assert_eq!(roots.root_operations(), &[inserted, existing]);

    let mut nested = generic("\"container\"() ({ ^entry: \"child\"() : () -> () }) : () -> ()");
    let container = nested.root_operations()[0];
    let region = nested.operation_regions(container).unwrap()[0];
    let block = nested.region(region).unwrap().blocks(&nested).unwrap()[0];
    let mut editor = nested.edit(DialectRegistry::proving()).unwrap();
    let inserted = editor
        .insert(
            InsertionPoint::Block { block, index: 1 },
            unknown_spec("nested.inserted"),
        )
        .unwrap();
    editor.commit().unwrap();
    validate(&nested);
    assert_eq!(nested.block_operations(block).unwrap()[1], inserted);
    assert_eq!(
        nested.operation(inserted).unwrap().parent_block(),
        Some(block)
    );
}

#[test]
fn attributes_properties_foreign_values_and_incomplete_documents_are_bounded() {
    let mut document = generic("\"editable\"() : () -> ()");
    let operation = document.root_operations()[0];
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor
        .set_attribute(
            operation,
            AttributeSpec {
                name: "tag".into(),
                spelling: "\"new\"".into(),
                value: zirium::semantic::AttributeValue::String("\"new\"".into()),
            },
        )
        .unwrap();
    editor
        .set_property(
            operation,
            AttributeSpec {
                name: "flag".into(),
                spelling: "1".into(),
                value: zirium::semantic::AttributeValue::Integer("1".into()),
            },
        )
        .unwrap();
    editor.commit().unwrap();
    validate(&document);
    assert_eq!(
        document.attributes(operation).unwrap().next(),
        Some(("tag", "\"new\""))
    );
    assert_eq!(
        document.properties(operation).unwrap().next(),
        Some(("flag", "1"))
    );

    let other = generic("%x = \"foreign\"() : () -> i32");
    let foreign = other.root_operations()[0];
    let foreign_value = other
        .operation(foreign)
        .unwrap()
        .result(foreign, 0)
        .unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    let mut spec = unknown_spec("bad.insert");
    spec.operands.push(foreign_value);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(0), spec),
        Err(EditError::ForeignValue(value)) if value == foreign_value
    ));

    let parsed = ParsedFile::parse(b"%x = \"bad\"(%missing) : (i32) -> i32".as_slice()).unwrap();
    let mut incomplete = lower_proving_fixture(&parsed, LoweringMode::BestEffort, &SharedRegistry)
        .document
        .unwrap();
    assert!(matches!(
        incomplete.edit(DialectRegistry::proving()),
        Err(EditError::IncompleteDocument)
    ));
}

#[test]
fn successor_argument_rewire_allows_definition_deletion() {
    let mut document = registered(
        r#"%function = "func.func"() ({
^entry:
  %a = arith.constant 1 : i32
  %b = arith.constant 2 : i32
  cf.br ^exit(%a : i32)
^exit(%arg: i32):
  func.return %arg : i32
}) : () -> i32"#,
    );
    let constants = document
        .operations()
        .filter(|id| document.operation_name(*id) == Some("arith.constant"))
        .collect::<Vec<_>>();
    let branch = document
        .operations()
        .find(|id| document.operation_name(*id) == Some("cf.br"))
        .unwrap();
    let old_value = document
        .operation(constants[0])
        .unwrap()
        .result(constants[0], 0)
        .unwrap();
    let replacement = document
        .operation(constants[1])
        .unwrap()
        .result(constants[1], 0)
        .unwrap();

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert_eq!(editor.replace_all_uses(old_value, replacement).unwrap(), 1);
    editor.erase(constants[0]).unwrap();
    editor.commit().unwrap();

    validate(&document);
    let successor = document.successors(branch).unwrap()[0];
    assert_eq!(
        document.successor_arguments(successor).unwrap(),
        &[zirium::semantic::ValueReference::Resolved(replacement)]
    );
    assert!(document.operation(constants[0]).is_none());
    assert!(matches!(old_value, ValueId::OperationResult { .. }));
}

#[test]
fn replace_all_uses_rejects_type_changes_and_indexes_successor_arguments() {
    let mut document =
        generic("%a = \"a\"() : () -> i32\n%b = \"b\"() : () -> i64\n\"use\"(%a) : (i32) -> ()");
    let operations = document.operations().collect::<Vec<_>>();
    let from = document
        .operation(operations[0])
        .unwrap()
        .result(operations[0], 0)
        .unwrap();
    let to = document
        .operation(operations[1])
        .unwrap()
        .result(operations[1], 0)
        .unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert_eq!(
        editor.replace_all_uses(from, to),
        Err(EditError::TypeMismatch)
    );
}

#[test]
fn lazy_indexes_replace_uses_and_rebuild_after_revision() {
    let mut document = registered(
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b : i32",
    );
    let operations = document.operations().collect::<Vec<_>>();
    let from = document
        .operation(operations[0])
        .unwrap()
        .result(operations[0], 0)
        .unwrap();
    let to = document
        .operation(operations[1])
        .unwrap()
        .result(operations[1], 0)
        .unwrap();
    assert_eq!(document.statistics().use_index_entries, 0);
    assert_eq!(document.uses(from).len(), 1);
    assert_eq!(document.statistics().use_index_entries, 2);

    let old_revision = document.revision();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert_eq!(editor.replace_all_uses(from, to).unwrap(), 1);
    editor.erase(operations[0]).unwrap();
    editor.commit().unwrap();

    assert_eq!(document.revision(), old_revision + 1);
    assert!(document.uses(from).is_empty());
    assert_eq!(document.uses(to).len(), 2);
    validate(&document);
}

#[test]
fn symbol_and_dominance_indexes_follow_registered_descriptors() {
    let document = registered_best_effort(
        r#"builtin.module {
  func.func @callee() { func.return }
  func.func @caller() {
    %a = arith.constant 1 : i32
    %b = arith.constant 2 : i32
    %c = arith.addi %a, %b : i32
    func.call @callee() : () -> ()
    "unknown.reference"() {callee = @missing} : () -> ()
    func.return
  }
}"#,
    );
    let caller = document
        .operations()
        .find(|operation| {
            document.operation_name(*operation) == Some("func.func")
                && document
                    .attributes(*operation)
                    .unwrap()
                    .any(|(name, value)| name == "sym_name" && value.trim_matches('@') == "caller")
        })
        .unwrap();
    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.call"))
        .unwrap();
    let callee = document
        .lookup_symbol(call, "@callee", DialectRegistry::proving())
        .unwrap();
    assert_eq!(document.operation_name(callee), Some("func.func"));
    assert!(
        document
            .symbol_index_diagnostics(DialectRegistry::proving())
            .is_empty()
    );
    assert!(document.statistics().symbol_index_entries >= 2);

    let constants = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("arith.constant"))
        .collect::<Vec<_>>();
    let add = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arith.addi"))
        .unwrap();
    let first = document
        .operation(constants[0])
        .unwrap()
        .result(constants[0], 0)
        .unwrap();
    assert!(document.dominates(first, add, DialectRegistry::proving()));
    assert!(!document.dominates(first, caller, DialectRegistry::proving()));
    assert!(document.statistics().dominance_index_entries > 0);
}

#[test]
fn registered_symbol_indexes_shadow_nested_scopes_and_report_unresolved_refs() {
    let document = registered_best_effort(
        r#"builtin.module {
  func.func @f() { func.return }
  func.func @outer() { func.call @f() : () -> () func.return }
  builtin.module {
    func.func @f() { func.return }
    func.func @inner() {
      func.call @f() : () -> ()
      func.call @missing() : () -> ()
      func.return
    }
  }
}"#,
    );
    let calls = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("func.call"))
        .collect::<Vec<_>>();
    let outer_target = document.lookup_symbol(calls[0], "@f", DialectRegistry::proving());
    let inner_target = document.lookup_symbol(calls[1], "@f", DialectRegistry::proving());
    assert!(outer_target.is_some() && inner_target.is_some());
    assert_ne!(outer_target, inner_target);
    let diagnostics = document.symbol_index_diagnostics(DialectRegistry::proving());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.symbol == "missing")
    );
}

#[test]
fn concurrent_registry_queries_keep_matching_symbol_and_dominance_results() {
    let document = Arc::new(registered_best_effort(
        r#"builtin.module {
  func.func @f() { func.return }
  func.func @caller() {
    %value = arith.constant 1 : i32
    func.call @f() : () -> ()
    func.call @missing() : () -> ()
    func.return
  }
}"#,
    ));
    let call = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.call"))
        .unwrap();
    let constant = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("arith.constant"))
        .unwrap();
    let return_operation = document
        .operations()
        .filter(|operation| document.operation_name(*operation) == Some("func.return"))
        .last()
        .unwrap();
    let value = document
        .operation(constant)
        .unwrap()
        .result(constant, 0)
        .unwrap();

    std::thread::scope(|scope| {
        for proving in [true, false].into_iter().cycle().take(16) {
            let document = Arc::clone(&document);
            scope.spawn(move || {
                let registry = if proving {
                    DialectRegistry::proving()
                } else {
                    &DialectRegistry::EMPTY
                };
                assert_eq!(
                    document.lookup_symbol(call, "@f", registry).is_some(),
                    proving
                );
                assert_eq!(
                    document
                        .symbol_index_diagnostics(registry)
                        .iter()
                        .any(|diagnostic| diagnostic.symbol == "missing"),
                    proving
                );
                assert!(document.dominates(value, return_operation, registry));
            });
        }
    });
}

#[test]
fn indexed_block_argument_dominance_matches_cfg_expectations() {
    let document = registered_best_effort(
        r#"builtin.module {
  func.func @flow() -> i1 {
  ^entry(%entry: i1):
    %cond = arith.constant 1 : i1
    cf.cond_br %cond, ^left(%entry : i1), ^right(%entry : i1)
  ^left(%left_arg: i1):
    cf.br ^merge(%left_arg : i1)
  ^right(%right_arg: i1):
    cf.br ^merge(%right_arg : i1)
  ^merge(%merge_arg: i1):
    func.return %merge_arg : i1
  }
}"#,
    );
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.func"))
        .unwrap();
    let blocks = document
        .region(document.operation_regions(function).unwrap()[0])
        .unwrap()
        .blocks(&document)
        .unwrap()
        .to_vec();
    let entry_value = ValueId::BlockArgument {
        block: blocks[0],
        argument: 0,
    };
    let left_value = ValueId::BlockArgument {
        block: blocks[1],
        argument: 0,
    };
    let return_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    assert!(document.dominates(entry_value, return_operation, DialectRegistry::proving()));
    assert!(!document.dominates(left_value, return_operation, DialectRegistry::proving()));
    assert!(!document.dominates(
        ValueId::BlockArgument {
            block: blocks[0],
            argument: 99,
        },
        return_operation,
        DialectRegistry::proving()
    ));
    let forged_result = ValueId::OperationResult {
        operation: document
            .operations()
            .find(|operation| document.operation_name(*operation) == Some("arith.constant"))
            .unwrap(),
        result: 99,
    };
    assert!(!document.dominates(forged_result, return_operation, DialectRegistry::proving()));
    assert!(document.statistics().dominance_index_entries > 0);
}

#[test]
fn editor_commit_rejects_invisible_successor_argument_atomically() {
    let mut document = registered(
        r#"builtin.module {
  func.func @flow() -> i1 {
  ^entry:
    %cond = arith.constant 1 : i1
    cf.cond_br %cond, ^left(%cond : i1), ^right(%cond : i1)
  ^left(%left_arg: i1):
    cf.br ^merge(%left_arg : i1)
  ^right(%right_arg: i1):
    cf.br ^merge(%right_arg : i1)
  ^merge(%merge_arg: i1):
    func.return %merge_arg : i1
  }
}"#,
    );
    let function = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.func"))
        .unwrap();
    let blocks = document
        .region(document.operation_regions(function).unwrap()[0])
        .unwrap()
        .blocks(&document)
        .unwrap()
        .to_vec();
    let left_branch = document
        .block_operations(blocks[1])
        .unwrap()
        .iter()
        .copied()
        .find(|operation| document.operation_name(*operation) == Some("cf.br"))
        .unwrap();
    let original = document
        .successor_arguments(document.successors(left_branch).unwrap()[0])
        .unwrap()[0];
    let right_value = ValueId::BlockArgument {
        block: blocks[2],
        argument: 0,
    };

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor
        .rewire_successor_argument(left_branch, 0, 0, right_value)
        .unwrap();
    assert!(matches!(
        editor.commit(),
        Err(EditError::Semantic(SemanticVerificationError::Operation {
            message,
            ..
        })) if message == "SSA definition does not dominate its use"
    ));
    assert_eq!(
        document
            .successor_arguments(document.successors(left_branch).unwrap()[0])
            .unwrap()[0],
        original
    );

    let return_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    let left_value = ValueId::BlockArgument {
        block: blocks[1],
        argument: 0,
    };
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor
        .rewire_operand(return_operation, 0, left_value)
        .unwrap();
    assert!(matches!(
        editor.commit(),
        Err(EditError::Semantic(SemanticVerificationError::Operation {
            message,
            ..
        })) if message == "SSA definition does not dominate its use"
    ));
}

#[test]
fn indexed_dominance_matches_full_verification_for_loops_and_unreachable_blocks() {
    let loop_document = registered_best_effort(
        r#"builtin.module {
  func.func @loop() {
  ^entry:
    %value = arith.constant 1 : i32
    %condition = arith.constant 1 : i1
    cf.br ^loop
  ^loop:
    "use"(%value) : (i32) -> ()
    cf.cond_br %condition, ^loop, ^exit
  ^exit:
    func.return
  }
}"#,
    );
    let value_operation = loop_document
        .operations()
        .find(|operation| loop_document.operation_name(*operation) == Some("arith.constant"))
        .unwrap();
    let value = loop_document
        .operation(value_operation)
        .unwrap()
        .result(value_operation, 0)
        .unwrap();
    let use_operation = loop_document
        .operations()
        .find(|operation| loop_document.operation_name(*operation) == Some("use"))
        .unwrap();
    assert!(
        loop_document
            .verify_semantics(DialectRegistry::proving())
            .is_ok()
    );
    assert!(loop_document.dominates(value, use_operation, DialectRegistry::proving()));

    let unreachable_document = registered_best_effort(
        r#"builtin.module {
  func.func @unreachable() {
  ^entry:
    %value = arith.constant 1 : i32
    cf.br ^exit
  ^dead:
    "use"(%value) : (i32) -> ()
    cf.br ^dead
  ^exit:
    func.return
  }
}"#,
    );
    let value_operation = unreachable_document
        .operations()
        .find(|operation| unreachable_document.operation_name(*operation) == Some("arith.constant"))
        .unwrap();
    let value = unreachable_document
        .operation(value_operation)
        .unwrap()
        .result(value_operation, 0)
        .unwrap();
    let use_operation = unreachable_document
        .operations()
        .find(|operation| unreachable_document.operation_name(*operation) == Some("use"))
        .unwrap();
    assert!(
        unreachable_document
            .verify_semantics(DialectRegistry::proving())
            .is_ok()
    );
    assert!(unreachable_document.dominates(value, use_operation, DialectRegistry::proving()));
    let statistics = unreachable_document.statistics();
    assert!(statistics.dominance_index_entries < statistics.operations + statistics.blocks * 20);
}

#[test]
fn dominance_index_preserves_graph_and_unregistered_ssacfg_registry_semantics() {
    fn unused_parser(
        _parser: &mut zirium::parser::DialectParser<'_, '_>,
    ) -> Result<(), zirium::CompactError> {
        unreachable!("the descriptor is used only for dominance metadata")
    }
    let operations = Box::leak(Box::new([zirium::dialect::OperationDescriptor {
        name: "outer",
        syntax_kind: zirium::SyntaxKind::DialectOperation,
        parse: Some(unused_parser),
        lower: None,
        verify: None,
        print: None,
        assembly: None,
        schema: zirium::dialect::OperationSchema {
            operands: zirium::dialect::OperandCount::Variadic,
            results: zirium::dialect::ResultCount::Variadic,
            required_attributes: &[],
        },
        regions: &[zirium::dialect::RegionDescriptor {
            kind: zirium::dialect::RegionKind::Graph,
            isolated_from_above: false,
        }],
        symbols: Default::default(),
    }]));
    let graph_registry = zirium::dialect::DialectRegistry::new(operations, &[], &[]);
    let document = generic_best_effort(
        r#""outer"() ({
  "use"(%value) : (i32) -> ()
  %value = "def"() : () -> i32
}) : () -> ()"#,
    );
    let definition = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("def"))
        .unwrap();
    let value = document
        .operation(definition)
        .unwrap()
        .result(definition, 0)
        .unwrap();
    let use_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("use"))
        .unwrap();
    assert_eq!(document.statistics().dominance_index_entries, 0);
    document.verify_semantics(&graph_registry).unwrap();
    assert_eq!(document.statistics().dominance_index_entries, 0);
    assert!(matches!(
        document.verify_semantics(&DialectRegistry::EMPTY),
        Err(SemanticVerificationError::Operation { message, .. })
            if message == "SSA definition does not dominate its use"
    ));
    assert_eq!(document.statistics().dominance_index_entries, 0);
    assert!(document.dominates(value, use_operation, &graph_registry));
    assert!(!document.dominates(value, use_operation, &DialectRegistry::EMPTY));

    let mut block_document = generic(
        r#""outer"() ({
^entry(%seed: i32):
  "branch"() [^left : (%seed : i32), ^right : (%seed : i32)] : () -> ()
^left(%arg: i32):
  "use_left"(%arg) : (i32) -> ()
^right(%other: i32):
  "use_right"(%other) : (i32) -> ()
}) : () -> ()"#,
    );
    let outer = block_document.root_operations()[0];
    let blocks = block_document
        .region(block_document.operation_regions(outer).unwrap()[0])
        .unwrap()
        .blocks(&block_document)
        .unwrap();
    let argument = ValueId::BlockArgument {
        block: blocks[1],
        argument: 0,
    };
    let use_operation = block_document
        .operations()
        .find(|operation| block_document.operation_name(*operation) == Some("use_right"))
        .unwrap();
    assert!(!block_document.dominates(argument, use_operation, &graph_registry));
    let mut editor = block_document.edit(&graph_registry).unwrap();
    editor.rewire_operand(use_operation, 0, argument).unwrap();
    assert!(matches!(
        editor.commit(),
        Err(EditError::Semantic(SemanticVerificationError::Operation {
            message,
            ..
        })) if message == "SSA definition does not dominate its use"
    ));
}

#[test]
fn committed_edit_invalidates_and_rebuilds_the_dominance_index() {
    let mut document = registered(
        "func.func @f() { %value = arith.constant 1 : i32 \"use\"(%value) : (i32) -> () func.return }",
    );
    let operations = document
        .operations()
        .filter(|operation| {
            matches!(
                document.operation_name(*operation),
                Some("arith.constant" | "use")
            )
        })
        .collect::<Vec<_>>();
    let value = document
        .operation(operations[0])
        .unwrap()
        .result(operations[0], 0)
        .unwrap();
    assert!(document.dominates(value, operations[1], DialectRegistry::proving()));
    assert!(document.statistics().dominance_index_entries > 0);

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor
        .set_attribute(
            operations[1],
            AttributeSpec {
                name: "tag".into(),
                spelling: "1 : i32".into(),
                value: zirium::semantic::AttributeValue::Integer("1".into()),
            },
        )
        .unwrap();
    editor.commit().unwrap();
    assert_eq!(document.statistics().dominance_index_entries, 0);
    assert!(document.dominates(value, operations[1], DialectRegistry::proving()));
    assert!(document.statistics().dominance_index_entries > 0);
}

#[test]
fn nested_block_argument_visibility_uses_enclosing_cfg_block() {
    let document = generic_best_effort(
        r#""outer"() ({
^entry(%arg: i32):
  "container"() ({
    "use"(%arg) : (i32) -> ()
  }) : () -> ()
}) : () -> ()"#,
    );
    let outer = document.root_operations()[0];
    let outer_block = document
        .region(document.operation_regions(outer).unwrap()[0])
        .unwrap()
        .blocks(&document)
        .unwrap()[0];
    let use_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("use"))
        .unwrap();
    let value = ValueId::BlockArgument {
        block: outer_block,
        argument: 0,
    };
    assert!(document.dominates(value, use_operation, DialectRegistry::proving()));
}

#[test]
fn isolated_nested_region_rejects_outer_block_argument_capture() {
    let document = registered_best_effort(
        r#""outer"() ({
^entry(%arg: i32):
  func.func @inner() -> i32 {
    func.return %arg : i32
  }
}) : () -> ()"#,
    );
    let outer = document.root_operations()[0];
    let outer_block = document
        .region(document.operation_regions(outer).unwrap()[0])
        .unwrap()
        .blocks(&document)
        .unwrap()[0];
    let return_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("func.return"))
        .unwrap();
    let value = ValueId::BlockArgument {
        block: outer_block,
        argument: 0,
    };
    assert!(!document.dominates(value, return_operation, DialectRegistry::proving()));
}

#[test]
fn pool_compaction_reclaims_fragments_without_changing_live_ids() {
    let mut document = registered(
        "%a = arith.constant 1 : i32\n%b = arith.constant 2 : i32\n%c = arith.addi %a, %b : i32",
    );
    let operations = document.operations().collect::<Vec<_>>();
    let first = document
        .operation(operations[0])
        .unwrap()
        .result(operations[0], 0)
        .unwrap();
    let second = document
        .operation(operations[1])
        .unwrap()
        .result(operations[1], 0)
        .unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    for value in [second, first, second, first, second] {
        editor.rewire_operand(operations[2], 0, value).unwrap();
    }
    let before = editor.document().statistics().pooled_list_entries;
    let reclaimed = editor.compact_pools();
    assert!(reclaimed > 0);
    assert!(editor.document().statistics().pooled_list_entries < before);
    editor.commit().unwrap();

    for operation in operations {
        assert!(document.operation(operation).is_some());
    }
    validate(&document);
}

#[test]
fn pool_compaction_preserves_nested_entities_and_stales_erased_operations() {
    let mut document = generic(include_str!(
        "../../../tests/corpus/mlir-22.1/generic-complete/valid.mlir"
    ));
    let container = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.regions"))
        .unwrap();
    let region = document.operation_regions(container).unwrap()[0];
    let block = document.region(region).unwrap().blocks(&document).unwrap()[1];
    let child = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.in_block"))
        .unwrap();
    let live_operation = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.results"))
        .unwrap();
    let live_value = document
        .operation(live_operation)
        .unwrap()
        .result(live_operation, 0)
        .unwrap();
    let child_type = document.function_type(child).unwrap();
    let properties = document
        .operations()
        .find(|operation| document.operation_name(*operation) == Some("test.properties"))
        .unwrap();
    let property = document.attribute_id(properties, "discardable").unwrap();
    let before = document.statistics().pooled_list_entries;

    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    let dead = editor
        .document()
        .operations()
        .find(|operation| editor.document().operation_name(*operation) == Some("test.empty"))
        .unwrap();
    editor.erase(dead).unwrap();
    assert!(editor.compact_pools() < before);

    assert!(editor.document().operation(dead).is_none());
    assert!(editor.document().operation(container).is_some());
    assert!(editor.document().region(region).is_some());
    assert!(editor.document().block(block).is_some());
    assert_eq!(editor.document().function_type(child), Some(child_type));
    assert_eq!(
        editor.document().attribute_id(properties, "discardable"),
        Some(property)
    );
    assert_eq!(
        editor.document().operation_location(child),
        Some(Some("loc(\"operation\")"))
    );
    assert_eq!(
        editor
            .document()
            .operation(live_operation)
            .unwrap()
            .result(live_operation, 0),
        Some(live_value)
    );
    editor.document().validate_structure().unwrap();
}

#[test]
fn stale_invalid_and_foreign_handles_have_distinct_edit_errors() {
    let mut document = generic("%x = \"value\"() : () -> i32");
    let local = document.root_operations()[0];
    let local_value = document.operation(local).unwrap().result(local, 0).unwrap();
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    let invalid = ValueId::OperationResult {
        operation: local,
        result: 99,
    };
    let mut spec = unknown_spec("invalid.value");
    spec.operands.push(invalid);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(0), spec),
        Err(EditError::InvalidValue(value)) if value == invalid
    ));
    editor.erase(local).unwrap();
    assert!(matches!(
        editor.erase(local),
        Err(EditError::StaleOperation(id)) if id == local
    ));
    let mut spec = unknown_spec("stale.value");
    spec.operands.push(local_value);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(0), spec),
        Err(EditError::StaleValue(value)) if value == local_value
    ));

    let stale = ValueId::OperationResult {
        operation: local,
        result: 99,
    };
    let mut spec = unknown_spec("invalid.value");
    spec.operands.push(stale);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(0), spec),
        Err(EditError::StaleValue(value)) if value == stale
    ));

    let other = generic("%y = \"other\"() : () -> i32");
    let foreign_operation = other.root_operations()[0];
    let foreign_value = other
        .operation(foreign_operation)
        .unwrap()
        .result(foreign_operation, 0)
        .unwrap();
    assert!(matches!(
        editor.erase(foreign_operation),
        Err(EditError::ForeignOperation(id)) if id == foreign_operation
    ));
    let mut spec = unknown_spec("foreign.value");
    spec.operands.push(foreign_value);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(0), spec),
        Err(EditError::ForeignValue(value)) if value == foreign_value
    ));
}

#[test]
fn hybrid_edits_invalidate_syntax_retention_before_commit() {
    let parsed = ParsedFile::parse(b"\"original\"() : () -> ()".as_slice()).unwrap();
    let mut document = lower_proving_fixture_with_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::Hybrid,
        &SharedRegistry,
    )
    .document
    .unwrap();
    assert_eq!(document.retention_profile(), RetentionProfile::Hybrid);
    assert!(
        document
            .operation_syntax_range(document.root_operations()[0])
            .is_some()
    );

    let original = document.root_operations()[0];
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    editor
        .insert(InsertionPoint::Root(0), unknown_spec("inserted"))
        .unwrap();
    editor.erase(original).unwrap();
    editor.commit().unwrap();

    validate(&document);
    assert_eq!(document.retention_profile(), RetentionProfile::SemanticOnly);
    assert!(document.source_bytes().is_none());
    assert!(document.syntax_tree().is_none());
    assert!(
        document
            .root_operations()
            .iter()
            .all(|id| { document.operation_syntax_range(*id).is_none() })
    );
}

#[test]
fn invalid_insertion_position_does_not_orphan_staged_operation() {
    let mut document = generic("\"original\"() : () -> ()");
    let mut editor = document.edit(DialectRegistry::proving()).unwrap();
    assert_eq!(
        editor.insert(InsertionPoint::Root(usize::MAX), unknown_spec("invalid")),
        Err(EditError::InvalidPosition)
    );
    let inserted = editor
        .insert(InsertionPoint::Root(1), unknown_spec("valid"))
        .unwrap();
    editor.commit().unwrap();
    validate(&document);
    assert!(document.operation(inserted).is_some());
    assert_eq!(document.root_operations().len(), 2);
}

#[test]
fn provisional_handles_from_dropped_or_failed_transactions_are_stale() {
    let mut dropped = generic("\"original\"() : () -> ()");
    let dropped_operation = {
        let mut editor = dropped.edit(DialectRegistry::proving()).unwrap();
        editor
            .insert(
                InsertionPoint::Root(1),
                OperationSpec {
                    result_types: vec![i32_type()],
                    ..unknown_spec("dropped")
                },
            )
            .unwrap()
    };
    let mut editor = dropped.edit(DialectRegistry::proving()).unwrap();
    assert!(matches!(
        editor.erase(dropped_operation),
        Err(EditError::StaleOperation(id)) if id == dropped_operation
    ));

    let mut failed = registered("%constant = arith.constant 1 : i32");
    let original = failed.root_operations()[0];
    let (failed_operation, failed_value) = {
        let mut editor = failed.edit(DialectRegistry::proving()).unwrap();
        let operation = editor
            .insert(
                InsertionPoint::Root(1),
                OperationSpec {
                    result_types: vec![i32_type()],
                    ..unknown_spec("failed")
                },
            )
            .unwrap();
        let value = editor
            .document()
            .operation(operation)
            .unwrap()
            .result(operation, 0)
            .unwrap();
        editor.remove_attribute(original, "value").unwrap();
        assert!(matches!(editor.commit(), Err(EditError::Semantic(_))));
        (operation, value)
    };
    let mut editor = failed.edit(DialectRegistry::proving()).unwrap();
    assert!(matches!(
        editor.erase(failed_operation),
        Err(EditError::StaleOperation(id)) if id == failed_operation
    ));
    let mut spec = unknown_spec("failed.value");
    spec.operands.push(failed_value);
    assert!(matches!(
        editor.insert(InsertionPoint::Root(1), spec),
        Err(EditError::StaleValue(value)) if value == failed_value
    ));
    validate(&failed);
}

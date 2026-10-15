use zirium::{
    dialect::DialectRegistry,
    diff::{ChangeField, DiffLimits, DiffOptions, compare},
    parser::ParsedFile,
    query::{changed, changes, dialect, diff_op_input, op},
    semantic::{Document, LoweringMode, lower_with_dialect_registry},
};

fn document(source: &str) -> Document {
    let registry = DialectRegistry::baseline();
    let parsed = ParsedFile::parse_with_registry(source.as_bytes(), registry).unwrap();
    lower_with_dialect_registry(&parsed, LoweringMode::Strict, registry)
        .document
        .unwrap()
}

#[test]
fn typed_change_filters_project_and_navigate_on_the_after_document() {
    let before =
        document("module { %value = arith.constant 4 : i32 \"test.use\"(%value) : (i32) -> () }");
    let after = document(
        "module { %renamed = arith.constant 8 : i32 \"test.use\"(%renamed) : (i32) -> () }",
    );
    let comparison = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();

    assert!(
        comparison
            .query(&changes().filter(op("missing")).names().min())
            .unwrap()
            .is_empty()
    );

    let rewired = changes().filter(changed(ChangeField::Attributes) & dialect("arith"));
    assert_eq!(comparison.query(&rewired).unwrap().len(), 1);

    let consumers = rewired.after().users().unique().names();
    assert_eq!(comparison.query(&consumers).unwrap(), ["test.use"]);
    assert_eq!(comparison.query(&rewired.before().count()).unwrap(), 1);
    assert_eq!(comparison.query(&rewired.attr("value")).unwrap(), ["8"]);
    assert_eq!(comparison.query(&rewired.result_types()).unwrap(), ["i32"]);
    assert_eq!(
        comparison
            .query(&rewired.after().users().operand_types())
            .unwrap(),
        ["i32"]
    );
    assert_eq!(
        comparison.query(&changes().names().sort().min()).unwrap(),
        ["arith.constant"]
    );

    let roots = rewired.after().root(op("builtin.module")).unique();
    assert_eq!(comparison.query(&roots).unwrap().len(), 1);
    assert_eq!(
        comparison
            .query(
                &rewired
                    .after()
                    .root(op("builtin.module"))
                    .subtree()
                    .filter(op("test.use"))
                    .names(),
            )
            .unwrap(),
        ["test.use"]
    );
    assert_eq!(
        comparison
            .query(&changes().reverse().head(1).tail(1).count())
            .unwrap(),
        1
    );
}

#[test]
fn typed_projected_graph_traversals_use_the_selected_document() {
    let before = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let after = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %y, %x : i32 }",
    );
    let comparison = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let changed_add = changes().filter(changed(ChangeField::Operands)).after();
    let expected = ["arith.constant", "arith.constant", "arith.addi"];
    assert_eq!(
        comparison.query(&changed_add.slice().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison.query(&changed_add.closure().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison.query(&changed_add.reachable().names()).unwrap(),
        expected
    );
    assert_eq!(
        comparison
            .query(&changed_add.fixpoint(&diff_op_input().closure()).names())
            .unwrap(),
        expected
    );
}

#[test]
fn text_programs_keep_native_side_results_and_the_typed_projection_contract() {
    use zirium::{
        diff::DiffSide,
        query::{DiffProgram, DiffProgramOutput, EvaluationLimits, EvaluationOptions},
    };
    let before = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let after = document(
        "module { %x = arith.constant 4 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let typed = diff
        .query(&changes().after().fixpoint(&diff_op_input().closure()))
        .unwrap();
    let mut emitted = Vec::new();
    DiffProgram::parse("selected = after | fixpoint(closure); selected")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                emitted.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        emitted,
        [DiffProgramOutput::Operations(
            typed.side(),
            typed.operations().to_vec()
        )]
    );
    assert_eq!(typed.side(), DiffSide::After);
    let selected = diff.query(&changes().after()).unwrap();
    for (source, expected) in [
        (
            "after | reverse union after",
            selected.operations().to_vec(),
        ),
        (
            "after | reverse intersect after",
            selected.operations().to_vec(),
        ),
        ("after except after", vec![]),
    ] {
        let mut actual = Vec::new();
        DiffProgram::parse(source)
            .unwrap()
            .evaluate(
                &diff,
                ("before", "after"),
                EvaluationLimits::default(),
                EvaluationOptions::default(),
                |_, value, _| {
                    actual.push(value);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            actual,
            [DiffProgramOutput::Operations(DiffSide::After, expected)]
        );
    }
    assert_eq!(
        diff.query(&changes().names().unique()).unwrap(),
        ["arith.constant"]
    );
    let error = DiffProgram::parse("names | unique")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unique requires a change or operation stream")
    );

    let nested_before = document(
        r#""test.root"() ({ "test.scope"() ({ "test.inner"() {tag = "old"} : () -> () }) : () -> () "test.sibling"() : () -> () }) : () -> ()"#,
    );
    let nested_after = document(
        r#""test.root"() ({ "test.scope"() ({ "test.inner"() {tag = "new"} : () -> () }) : () -> () "test.sibling"() : () -> () }) : () -> ()"#,
    );
    let nested = compare(
        &nested_before,
        &nested_after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    assert_eq!(
        nested
            .query(&changes().after().parent().parent().subtree().names())
            .unwrap(),
        ["test.root", "test.scope", "test.sibling", "test.inner"]
    );
    let mut names = Vec::new();
    DiffProgram::parse("after | parent | parent | subtree | names")
        .unwrap()
        .evaluate(
            &nested,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                names.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        names,
        [DiffProgramOutput::Strings(vec![
            "test.root".into(),
            "test.scope".into(),
            "test.sibling".into(),
            "test.inner".into()
        ])]
    );

    emitted.clear();
    DiffProgram::parse("(filter(false) | before) union (filter(false) | before)")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                emitted.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        emitted,
        [DiffProgramOutput::Operations(DiffSide::Before, Vec::new())]
    );
    let error = DiffProgram::parse("(filter(false) | before) union (filter(false) | after)")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("same diff side"));
}

#[test]
fn nested_diff_bodies_share_work_stream_nesting_and_result_allowances() {
    use zirium::query::{DiffProgram, EvaluationLimits, EvaluationOptions};
    let before = document(
        "module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let after = document(
        "module { %x = arith.constant 4 : i32 %y = arith.constant 2 : i32 %z = arith.addi %x, %y : i32 }",
    );
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let roots = changes().after().root(op("builtin.module"));
    let growth = roots.fixpoint(&diff_op_input().subtree());
    assert!(
        growth
            .evaluate(
                &diff,
                EvaluationLimits {
                    max_work: 1000,
                    max_items: 6
                }
            )
            .unwrap_err()
            .to_string()
            .contains("stream size limit")
    );
    assert!(
        growth
            .evaluate(
                &diff,
                EvaluationLimits {
                    max_work: 20,
                    max_items: 100
                }
            )
            .unwrap_err()
            .to_string()
            .contains("work limit")
    );
    let bounded = roots.fixpoint(&diff_op_input().subtree().unique());
    assert_eq!(
        bounded
            .evaluate(
                &diff,
                EvaluationLimits {
                    max_work: 1000,
                    max_items: 8
                }
            )
            .unwrap()
            .len(),
        4
    );

    let nested = DiffProgram::parse("after | fixpoint(fixpoint(filter(true)))").unwrap();
    let mut budget = zirql::Budget::new(zirql::Limits {
        work: 1000,
        items: 6,
        nesting: 1,
    });
    let error = nested
        .evaluate_with_budget(
            &diff,
            ("before", "after"),
            EvaluationOptions::default(),
            &mut budget,
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("nesting limit"));
    assert!(error.range().is_some());
    assert!(budget.remaining_work() < 1000);
    let mut budget = zirql::Budget::new(zirql::Limits {
        work: 1000,
        items: 6,
        nesting: 8,
    });
    nested
        .evaluate_with_budget(
            &diff,
            ("before", "after"),
            EvaluationOptions::default(),
            &mut budget,
            |_, _, _| Ok(()),
        )
        .unwrap();

    let traversal = DiffProgram::parse("after | fixpoint(closure)").unwrap();
    let mut budget = zirql::Budget::for_program(zirql::ProgramLimits {
        work: 1000,
        results: 8,
        nesting: 8,
        stream_items: 6,
    });
    let error = traversal
        .evaluate_with_budget(
            &diff,
            ("before", "after"),
            EvaluationOptions::default(),
            &mut budget,
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("result limit"));
    let mut budget = zirql::Budget::for_program(zirql::ProgramLimits {
        work: 1000,
        results: 100,
        nesting: 8,
        stream_items: 6,
    });
    traversal
        .evaluate_with_budget(
            &diff,
            ("before", "after"),
            EvaluationOptions::default(),
            &mut budget,
            |_, _, _| Ok(()),
        )
        .unwrap();

    let mut body = diff_op_input();
    for _ in 0..1000 {
        body = diff_op_input().fixpoint(&body);
    }
    assert!(
        diff.query(&changes().after().fixpoint(&body))
            .unwrap_err()
            .to_string()
            .contains("nesting limit")
    );
    let mut predicate = changed(ChangeField::Attributes);
    for _ in 0..1000 {
        predicate = !predicate;
    }
    assert!(
        diff.query(&changes().filter(predicate))
            .unwrap_err()
            .to_string()
            .contains("nesting limit")
    );
    let mut predicate = changed(ChangeField::Attributes);
    for _ in 0..128 {
        predicate = predicate | changed(ChangeField::Attributes);
    }
    assert_eq!(diff.query(&changes().filter(predicate).count()).unwrap(), 1);
    let source = format!(
        "after | {}filter(true){}",
        "fixpoint(".repeat(1000),
        ")".repeat(1000)
    );
    assert!(
        DiffProgram::parse(&source)
            .unwrap_err()
            .to_string()
            .contains("nesting limit")
    );
}

#[test]
fn diff_fixed_points_refuse_ordered_cycles_and_side_replacement() {
    use zirium::query::{DiffProgram, EvaluationLimits, EvaluationOptions};
    let before = document("module { %x = arith.constant 1 : i32 %y = arith.constant 2 : i32 }");
    let after = document("module { %x = arith.constant 4 : i32 %y = arith.constant 2 : i32 }");
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let cycle = changes()
        .after()
        .root(op("builtin.module"))
        .children()
        .fixpoint(&diff_op_input().reverse());
    assert!(
        diff.query(&cycle)
            .unwrap_err()
            .to_string()
            .contains("cycles")
    );
    for (source, message) in [
        (
            "after | root(op(\"builtin.module\")) | children | fixpoint(reverse)",
            "cycles",
        ),
        (
            "after | fixpoint(input | before)",
            "preserve the projected diff side",
        ),
    ] {
        let error = DiffProgram::parse(source)
            .unwrap()
            .evaluate(
                &diff,
                ("before", "after"),
                EvaluationLimits::default(),
                EvaluationOptions::default(),
                |_, _, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert!(error.range().is_some());
    }
}

#[test]
fn flat_diff_sets_parse_and_drop_on_a_normal_thread() {
    std::thread::spawn(|| {
        use zirium::query::DiffProgram;
        let flat = vec!["filter(false)"; 25_000].join(" union ");
        // Both the program's roots and nested host expressions own flat spines.
        let parsed = DiffProgram::parse(&format!("{flat}; do ({flat})")).unwrap();
        drop(parsed);
        // Later frontend failures also release partially accumulated expressions
        // and statements rather than falling back to recursive drop glue.
        assert!(DiffProgram::parse(&format!("{flat} union unknown()")).is_err());
        assert!(DiffProgram::parse(&format!("{flat}; unknown()")).is_err());
    })
    .join()
    .unwrap();
}

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
fn diff_reader_rejects_unused_stage_arguments() {
    use zirium::query::DiffProgram;

    for stage in [
        "head(1, 2)",
        "check(1, 2)",
        "reverse(undefined())",
        "unique()",
    ] {
        let source = format!("after | {stage} | count");
        let error = DiffProgram::parse(&source).unwrap_err();
        assert_eq!(error.range(), Some(8..8 + stage.len()), "{source}: {error}");
    }
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
            selected.operations().iter().rev().copied().collect(),
        ),
        (
            "after | reverse intersect after",
            selected.operations().iter().rev().copied().collect(),
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
    let mut maps = Vec::new();
    DiffProgram::parse("map_by(names, before | filter(op(\"missing\")))")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                maps.push(value);
                Ok(())
            },
        )
        .unwrap();
    let DiffProgramOutput::Map(entries) = &maps[0] else {
        panic!("native map")
    };
    assert!(
        matches!(&entries.iter().find(|(key, _)| key == "arith.constant").unwrap().1, DiffProgramOutput::Operations(DiffSide::Before, operations) if operations.is_empty())
    );
    maps.clear();
    DiffProgram::parse("map_by(names, after | map_by(names, count))")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                maps.push(value);
                Ok(())
            },
        )
        .unwrap();
    let DiffProgramOutput::Map(entries) = &maps[0] else {
        panic!("native map")
    };
    assert!(
        matches!(&entries.iter().find(|(key, _)| key == "arith.constant").unwrap().1, DiffProgramOutput::Map(nested) if nested.iter().find(|(key, _)| key == "arith.constant").unwrap().1 == DiffProgramOutput::Count(1))
    );

    assert_eq!(
        diff.query(&changes().names().unique()).unwrap(),
        ["arith.constant"]
    );
    let mut textual = Vec::new();
    DiffProgram::parse("names | unique")
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                textual.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        textual,
        [DiffProgramOutput::Strings(vec!["arith.constant".into()])]
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
        (
            "after | filter(false) | fixpoint(input | before | filter(false))",
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

#[test]
fn dictionaries_and_sets_retain_operand_order_and_native_sides() {
    use zirium::query::{
        DiffProgram, DiffProgramOutput, EvaluationLimits, EvaluationOptions, diff_value_to_json,
    };
    let before = document(
        r#"module {
        "test.z"() {v = 1 : i64} : () -> ()
        "test.a"() {v = 1 : i64} : () -> ()
        "test.m"() {v = 1 : i64} : () -> ()
    }"#,
    );
    let after = document(
        r#"module {
        "test.z"() {v = 2 : i64} : () -> ()
        "test.a"() {v = 2 : i64} : () -> ()
        "test.m"() {v = 2 : i64} : () -> ()
    }"#,
    );
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let source = r#"
        selected = reverse;
        counts = selected | map_by(names, count);
        counts;
        counts | sort_by(value) | reverse | sort_by( value ) | head(2);
        selected | map_by(names, before | map_by(names, count));
        counts | head(0);
        filter(false) | names | tally | sort_by(value) | reverse | tail(1);
        (selected union input) | names;
        (selected intersect input) | names;
        (selected except head(1)) | names;
        (after | parent union after | parent) | names;
        (after | reverse intersect after) | names;
        after | filter(false) | min_by(names);
    "#;
    let mut outputs = vec![];
    DiffProgram::parse(source)
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                outputs.push(value);
                Ok(())
            },
        )
        .unwrap();
    let mut budget = zirql::Budget::new(zirql::Limits {
        work: 100_000,
        items: 1_000,
        nesting: 64,
    });
    let json = diff_value_to_json(&diff, &outputs[0], &mut budget).unwrap();
    assert_eq!(
        json.as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["test.m", "test.a", "test.z"]
    );
    let json = diff_value_to_json(&diff, &outputs[1], &mut budget).unwrap();
    assert_eq!(
        json.as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["test.z", "test.a"]
    );
    let DiffProgramOutput::Map(nested) = &outputs[2] else {
        panic!("native dictionary");
    };
    let DiffProgramOutput::Map(inner) = &nested[0].1 else {
        panic!("nested dictionary");
    };
    assert_eq!(inner[0].0, "test.m");
    let json = diff_value_to_json(&diff, &outputs[2], &mut budget).unwrap();
    assert_eq!(
        json.as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["test.m", "test.a", "test.z"]
    );
    assert!(matches!(&outputs[3], DiffProgramOutput::Map(entries) if entries.is_empty()));
    assert!(
        matches!(&outputs[4], DiffProgramOutput::Dictionary(value) if value.entries().is_empty())
    );
    assert_eq!(
        outputs[5],
        DiffProgramOutput::Strings(vec!["test.m".into(), "test.a".into(), "test.z".into()])
    );
    assert_eq!(outputs[6], outputs[5]);
    assert_eq!(
        outputs[7],
        DiffProgramOutput::Strings(vec!["test.m".into(), "test.a".into()])
    );
    assert_eq!(
        outputs[8],
        DiffProgramOutput::Strings(vec!["builtin.module".into()])
    );
    assert_eq!(outputs[9], outputs[5]);
    assert!(
        matches!(&outputs[10], DiffProgramOutput::Operations(zirium::diff::DiffSide::After, operations) if operations.is_empty())
    );
}

#[test]
fn diff_existence_preserves_subjects_and_preflights_empty_sides() {
    use zirium::query::{DiffProgram, DiffProgramOutput, EvaluationLimits, EvaluationOptions};
    let before = document("module { %x = arith.constant 1 : i32 %y = arith.addi %x, %x : i32 }");
    let after = document("module { %x = arith.constant 4 : i32 %y = arith.addi %x, %x : i32 }");
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let selected = changes().after().users().defs();
    let expected = diff
        .query(&selected.where_exists(&diff_op_input().users()))
        .unwrap();
    assert_eq!(expected.len(), 4);
    assert_eq!(expected.operations()[0], expected.operations()[1]);
    let mut actual = vec![];
    DiffProgram::parse("after | users | defs | where_exists(users) | where_exists(emit)")
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
        [DiffProgramOutput::Operations(
            expected.side(),
            expected.operations().to_vec()
        )]
    );
    for source in [
        "after | filter(false) | where_exists(input | before)",
        "after | filter(false) | where_exists(names)",
        "after | where_exists(users | check(99))",
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
        assert!(error.range().is_some(), "{source}: {error}");
    }
}

#[test]
fn shared_diff_predicates_keep_side_candidates_and_written_admission() {
    use zirium::query::{DiffProgram, DiffProgramOutput, EvaluationLimits, EvaluationOptions};
    let before = document("module { %x = arith.constant 1 : i32 }");
    let after = document("module { %x = arith.constant 2 : i32 }");
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let source = r#"selected = after; selected | filter(p => op(one(p | names)) and one(_ | names)=="arith.constant") | root(ancestor => op("builtin.module") and one(ancestor | names)=="builtin.module") | names"#;
    let mut values = Vec::new();
    DiffProgram::parse(source)
        .unwrap()
        .evaluate(
            &diff,
            ("before", "after"),
            EvaluationLimits::default(),
            EvaluationOptions::default(),
            |_, value, _| {
                values.push(value);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        values,
        [DiffProgramOutput::Strings(vec!["builtin.module".into()])]
    );
    for source in [
        "after | filter(false) | filter(p => p | count)",
        "after | filter(false) | root(p => change(\"modified\"))",
        "after | filter((a,b) => true)",
    ] {
        let mut emitted = false;
        let error = DiffProgram::parse(source)
            .unwrap()
            .evaluate(
                &diff,
                ("before", "after"),
                EvaluationLimits::default(),
                EvaluationOptions::default(),
                |_, _, _| {
                    emitted = true;
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(error.range().is_some(), "{source}: {error}");
        assert!(!emitted);
    }
}

#[test]
fn diff_tally_keeps_native_sides_saved_lookup_and_dictionary_continuation() {
    use zirium::query::{DiffProgram, DiffProgramOutput, EvaluationLimits, EvaluationOptions};
    let before = document("module { %x = arith.constant 1 : i32 }");
    let after = document("module { %x = arith.constant 2 : i32 }");
    let diff = compare(
        &before,
        &after,
        DialectRegistry::baseline(),
        DiffOptions::default(),
        DiffLimits::default(),
    )
    .unwrap();
    let mut outputs = Vec::new();
    DiffProgram::parse("keys=after; hist=keys | tally; saved=hist; saved | reverse | head(1) | check(1) | sort_by(value); lookup(hist,keys | head(1)); a=before | head(1); b=after | head(1); {a:1,b:2} | count").unwrap().evaluate(&diff, ("before", "after"), EvaluationLimits::default(), EvaluationOptions::default(), |_, value, _| { outputs.push(value); Ok(()) }).unwrap();
    let DiffProgramOutput::Dictionary(dictionary) = &outputs[0] else {
        panic!("dictionary")
    };
    assert_eq!(dictionary.entries().len(), 1);
    assert!(matches!(
        &dictionary.entries()[0].0,
        DiffProgramOutput::Operations(zirium::diff::DiffSide::After, _)
    ));
    assert_eq!(outputs[1], DiffProgramOutput::Count(1));
    assert_eq!(outputs[2], DiffProgramOutput::Count(2));
}

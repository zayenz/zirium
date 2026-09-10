use zirium::query::Query;
use zirium::query::{
    lexer::{
        DiagnosticKind as LexDiagnosticKind, MAX_QUERY_BYTES, TokenKind, lex, query_size_supported,
    },
    parser::{Predicate, Program, Stage, parse, parse_with_nesting_limit},
};

fn initial_predicate(program: &Program) -> &Predicate {
    let Stage::Filter { predicate, .. } = &program.expression().first[0] else {
        panic!("expected filter")
    };
    predicate
}

#[test]
fn query_boundaries_and_edit_validation() {
    for source in [
        "",
        "# just a comment",
        "input",
        "emit",
        "emit | input",
        "count",
        "filter(true)",
        "input | filter(false) | subtree | emit",
        "filter(op(\"x\")) | root(op(\"func.func\")) | unique | attr(\"sym_name\") | json",
        "fixpoint(closure)",
        "fixpoint(closure | emit)",
        "(defs | emit) union users",
        "(defs union users) | filter(true)",
        "fixpoint(filter(true) union users)",
        "names | sort | reverse | head(10) | tail(3) | min | max",
        "sort_by(attr(\"rank\")) | min_by(children | count) | max_by(names)",
        r#"do filter(op("x")) | set_attr("tag", "hot"); emit"#,
        r#"do filter(op("x")) | set_attr("tag", "hot");"#,
    ] {
        Query::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    }
    for source in [
        "count | root",
        "(count) | defs",
        "fixpoint(count)",
        "fixpoint(set_attr(\"tag\", \"x\"))",
        "input union (filter(true) | count)",
        "input union (set_attr(\"tag\", \"x\"))",
        "select(op(\"x\"))",
        "filter(true) | union(op(\"x\"))",
        "filter(true) |",
        "()",
        "fixpoint()",
        "head(-1)",
        "tail()",
        "sort_by(set_attr(\"tag\", \"x\"))",
        "min_by(emit)",
    ] {
        assert!(Query::parse(source).is_err(), "{source}");
    }
    for source in [
        r#"set_attr("bad name", "hot")"#,
        r#"remove_attr("bad name")"#,
    ] {
        assert!(
            Query::parse(source)
                .unwrap_err()
                .to_string()
                .contains("dotted ASCII identifier")
        );
    }
    assert!(
        Query::parse("set_attr(\"tag\", \"hot\nvalue\")")
            .unwrap_err()
            .to_string()
            .contains("control characters")
    );
    Query::parse(r#"set_attr("tag", "quoted \"value\" \\ path")"#).unwrap();
}

#[test]
fn parser_distinguishes_do_statements_from_emitting_queries() {
    let parsed = parse(&lex(
        r#"do filter(op("x")) | set_attr("tag", "hot"); filter(has_attr("tag"))"#,
    ));
    let program = parsed.program().unwrap();
    assert!(matches!(
        program.statements.as_slice(),
        [zirium::query::parser::Statement::Do(_)]
    ));
    assert!(matches!(
        program.expression().first.as_slice(),
        [Stage::Filter { .. }]
    ));

    let missing_semicolon = Query::parse(r#"do set_attr("tag", "hot")"#).unwrap_err();
    assert!(missing_semicolon.message.contains("`;` after do statement"));
}

#[test]
fn lexer_records_token_kinds_ranges_and_string_failures() {
    let source = "op (\"a\\\"b\") |\nroot";
    let lexed = lex(source);
    let kinds = lexed
        .tokens()
        .iter()
        .map(|token| token.kind())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            TokenKind::Identifier,
            TokenKind::Trivia,
            TokenKind::LParen,
            TokenKind::String,
            TokenKind::RParen,
            TokenKind::Trivia,
            TokenKind::Pipe,
            TokenKind::Trivia,
            TokenKind::Identifier,
            TokenKind::Eof,
        ]
    );
    assert_eq!(lexed.tokens()[3].range().as_range(), 4..10);
    assert!(lexed.diagnostics().is_empty());

    let invalid = lex("\"bad\\q\" @");
    assert_eq!(
        invalid.diagnostics()[0].kind(),
        LexDiagnosticKind::InvalidEscape
    );
    assert_eq!(invalid.diagnostics()[0].range().as_range(), 4..6);
    assert_eq!(
        invalid.diagnostics()[1].kind(),
        LexDiagnosticKind::InvalidToken
    );
    let trailing_escape = lex("\"bad\\");
    assert!(
        trailing_escape
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.kind() == LexDiagnosticKind::InvalidEscape)
    );
}

#[test]
fn parser_builds_ranged_syntax_with_insignificant_whitespace() {
    let source = " filter ( op ( \"arith.addi\" ) )\n| set_attr ( \"tag\" , \"hot\" ) | count ";
    let lexed = lex(source);
    let parsed = parse(&lexed);
    assert!(parsed.diagnostics().is_empty());
    let program = parsed.program().unwrap();
    assert_eq!(program.range().as_range(), 1..source.len());
    assert!(
        matches!(initial_predicate(program), Predicate::Op { name, .. } if name == "arith.addi")
    );
    assert!(
        matches!(&program.expression().first[1..], [Stage::SetAttr { name, value, .. }, Stage::Count { .. }] if name == "tag" && value == "hot")
    );
}

#[test]
fn parser_builds_ranged_remove_attr_stage() {
    let source = r#"filter(op("x")) | remove_attr("analysis.tag")"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        &parsed.program().unwrap().expression().first[1..],
        [Stage::RemoveAttr { name, range }] if name == "analysis.tag" && range.as_range() == (18..source.len())
    ));
}

#[test]
fn parser_builds_ranged_relationship_stages() {
    let source = r#"filter(op("x")) | defs | users | parent | children | subtree | unique"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        &parsed.program().unwrap().expression().first[1..],
        [
            Stage::Defs { .. },
            Stage::Users { .. },
            Stage::Parent { .. },
            Stage::Children { .. },
            Stage::Subtree { .. },
            Stage::Unique { .. }
        ]
    ));
    assert_eq!(
        parsed.program().unwrap().expression().first[1]
            .range()
            .as_range(),
        18..22
    );
}

#[test]
fn parser_builds_ordering_reduction_and_bound_stages() {
    let source = r#"sort_by(attr("rank")) | reverse | head(10) | tail(3) | min_by(children | count) | max_by(names)"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        parsed.program().unwrap().expression().first.as_slice(),
        [
            Stage::SortBy { .. },
            Stage::Reverse { .. },
            Stage::Head { count: 10, .. },
            Stage::Tail { count: 3, .. },
            Stage::MinBy { .. },
            Stage::MaxBy { .. }
        ]
    ));
}

#[test]
fn parser_builds_root_projection_and_json_stages() {
    let source = r#"root(op("func.func")) | attr("sym_name") | json"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        parsed.program().unwrap().expression().first.as_slice(),
        [
            Stage::Root {
                predicate: Predicate::Op { name: operation, .. },
                ..
            },
            Stage::Attr { name, .. },
            Stage::Json { .. }
        ] if operation == "func.func" && name == "sym_name"
    ));
    assert!(Query::parse("root").is_err());
}

#[test]
fn pipes_bind_more_tightly_than_set_operators() {
    use zirium::query::parser::SetOperator;
    let parsed = parse(&lex("input | defs union users | parent except children"));
    let expression = parsed.program().unwrap().expression();
    assert_eq!(expression.first.len(), 2);
    assert!(matches!(expression.rest.as_slice(), [
        (SetOperator::Union, right), (SetOperator::Except, last)
    ] if matches!(right.as_slice(), [Stage::Users { .. }, Stage::Parent { .. }])
        && matches!(last.as_slice(), [Stage::Children { .. }])));
    let grouped = parse(&lex("(defs union users) | parent"));
    assert!(matches!(
        grouped.program().unwrap().expression().first.as_slice(),
        [Stage::Group { .. }, Stage::Parent { .. }]
    ));
}

#[test]
fn parser_builds_boolean_predicates_with_precedence_and_parentheses() {
    let parsed = parse(&lex(
        r#"filter(op("a") or has_attr("tag") and not string_attr_eq("state", "skip"))"#,
    ));
    let predicate = initial_predicate(parsed.program().unwrap());
    assert!(matches!(
        predicate,
        Predicate::Or { predicates, .. }
            if matches!(predicates.as_slice(), [Predicate::Op { name, .. }, Predicate::And { predicates, .. }]
                if name == "a" && matches!(predicates.as_slice(), [Predicate::HasAttr { .. }, Predicate::Not { predicate, .. }]
                    if matches!(predicate.as_ref(), Predicate::Attr { name, value, .. } if name == "state" && value == "skip")))
    ));

    let grouped = parse(&lex(r#"filter((op("a") or op("b")) and has_attr("tag"))"#));
    assert!(matches!(
        initial_predicate(grouped.program().unwrap()),
        Predicate::And { predicates, .. }
            if matches!(predicates.first(), Some(Predicate::Group { predicate, range })
                if matches!(predicate.as_ref(), Predicate::Or { .. }) && range.as_range() == (7..27))
    ));
}

#[test]
fn malformed_and_over_nested_predicates_are_diagnosed() {
    let missing = parse(&lex(r#"filter(string_attr_eq("tag" "value"))"#));
    assert_eq!(
        missing.diagnostics()[0].message(),
        "expected `,` in string_attr_eq"
    );

    let at_limit = parse_with_nesting_limit(&lex(r#"filter((op("x")))"#), 3);
    assert!(at_limit.diagnostics().is_empty());
    let beyond = parse_with_nesting_limit(&lex(r#"filter(((op("x"))))"#), 3);
    assert_eq!(
        beyond.diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
}

#[test]
fn unary_and_boolean_chain_complexity_is_stack_safe() {
    let at_limit = format!("filter({}op(\"x\"))", "not ".repeat(62));
    assert!(
        parse_with_nesting_limit(&lex(&at_limit), 64)
            .diagnostics()
            .is_empty()
    );
    let beyond = format!("filter({}op(\"x\"))", "not ".repeat(63));
    assert_eq!(
        parse_with_nesting_limit(&lex(&beyond), 64).diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
    let pathological = format!("filter({}op(\"x\"))", "not ".repeat(50_000));
    assert_eq!(
        parse(&lex(&pathological)).diagnostics()[0].message(),
        "query nesting limit exceeded"
    );

    for operator in [" or ", " and "] {
        let chain = format!(
            "filter({})",
            std::iter::repeat_n("op(\"x\")", 10_000)
                .collect::<Vec<_>>()
                .join(operator)
        );
        let parsed = parse(&lex(&chain));
        match (operator, initial_predicate(parsed.program().unwrap())) {
            (" or ", Predicate::Or { predicates, .. })
            | (" and ", Predicate::And { predicates, .. }) => {
                assert_eq!(predicates.len(), 10_000)
            }
            _ => panic!("expected a flat boolean chain"),
        }
    }
}

#[test]
fn query_nesting_is_bounded_including_modifiers_and_groups() {
    for wrapper in ["fixpoint(", "("] {
        let valid = format!("{}input{}", wrapper.repeat(63), ")".repeat(63));
        assert!(Query::parse(&valid).is_ok());
        let invalid = format!("{}input{}", wrapper.repeat(64), ")".repeat(64));
        assert!(
            Query::parse(&invalid)
                .unwrap_err()
                .message
                .contains("nesting limit")
        );
    }
    let lexed = lex("filter(op(\"x\"))");
    let parsed = parse_with_nesting_limit(&lexed, 1);
    assert_eq!(
        parsed.diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
    assert_eq!(parsed.diagnostics()[0].range().as_range(), 9..10);
}

#[test]
fn lexed_input_binds_the_parser_source_and_size_checks_do_not_wrap() {
    let source = String::from("filter(op(\"é\"))");
    let lexed = lex(&source);
    let parsed = parse(&lexed);
    assert!(parsed.diagnostics().is_empty());
    assert!(
        matches!(initial_predicate(parsed.program().unwrap()), Predicate::Op { name, .. } if name == "é")
    );

    assert!(query_size_supported(MAX_QUERY_BYTES));
    assert!(!query_size_supported(MAX_QUERY_BYTES.saturating_add(1)));
}

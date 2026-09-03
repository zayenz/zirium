use zirium::query::Query;
use zirium::query::{
    lexer::{
        DiagnosticKind as LexDiagnosticKind, MAX_QUERY_BYTES, TokenKind, lex, query_size_supported,
    },
    parser::{Predicate, Stage, parse, parse_with_nesting_limit},
};

#[test]
fn parses_operation_name_selection_and_reports_positions() {
    let query = Query::parse("select(op(\"arith.addi\"))").unwrap();
    assert_eq!(query.operation_name(), Some("arith.addi"));
    let error = Query::parse("select(op(arith.addi))").unwrap_err();
    assert!(error.position > 0);
    assert!(error.to_string().contains("expected"));

    let closure = Query::parse("select(op(\"arith.addi\")) | closure").unwrap();
    assert_eq!(closure.operation_name(), Some("arith.addi"));
    assert_eq!(
        Query::parse("select(op(\"arith.addi\") or op(\"arith.muli\"))")
            .unwrap()
            .operation_name(),
        None
    );
    Query::parse("select(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | root")
        .unwrap();
    Query::parse("select(op(\"arith.addi\")) | remove_attr(\"analysis.tag\") | root").unwrap();
    let error = Query::parse("select(op(\"arith.addi\")) | count | root").unwrap_err();
    assert!(error.to_string().contains("root requires a selection"));
    let error =
        Query::parse("select(op(\"arith.addi\")) | set_attr(\"bad name\", \"hot\")").unwrap_err();
    assert!(error.to_string().contains("dotted ASCII identifier"));
    let error = Query::parse("select(op(\"arith.addi\")) | remove_attr(\"bad name\")").unwrap_err();
    assert!(error.to_string().contains("dotted ASCII identifier"));
    let error =
        Query::parse("select(op(\"arith.addi\")) | set_attr(\"tag\", \"hot\nvalue\")").unwrap_err();
    assert!(error.to_string().contains("control characters"));
    Query::parse(r#"select(op("arith.addi")) | set_attr("tag", "quoted \"value\" \\ path")"#)
        .unwrap();
    assert!(Query::parse("select(op(\"arith.addi\")) | other").is_err());
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
    let source = " select ( op ( \"arith.addi\" ) )\n| set_attr ( \"tag\" , \"hot\" ) | count ";
    let lexed = lex(source);
    let parsed = parse(&lexed);
    assert!(parsed.diagnostics().is_empty());
    let program = parsed.program().unwrap();
    assert_eq!(program.range().as_range(), 1..source.len());
    assert!(matches!(program.predicate(), Predicate::Op { name, .. } if name == "arith.addi"));
    assert!(
        matches!(program.stages(), [Stage::SetAttr { name, value, .. }, Stage::Count { .. }] if name == "tag" && value == "hot")
    );
}

#[test]
fn parser_builds_ranged_remove_attr_stage() {
    let source = r#"select(op("x")) | remove_attr("analysis.tag")"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        parsed.program().unwrap().stages(),
        [Stage::RemoveAttr { name, range }] if name == "analysis.tag" && range.as_range() == (18..source.len())
    ));
}

#[test]
fn parser_builds_ranged_relationship_stages() {
    let source = r#"select(op("x")) | defs | users | parent | children"#;
    let parsed = parse(&lex(source));
    assert!(parsed.diagnostics().is_empty());
    assert!(matches!(
        parsed.program().unwrap().stages(),
        [
            Stage::Defs { .. },
            Stage::Users { .. },
            Stage::Parent { .. },
            Stage::Children { .. }
        ]
    ));
    assert_eq!(
        parsed.program().unwrap().stages()[0].range().as_range(),
        18..22
    );
}

#[test]
fn parser_builds_ranged_set_stages_with_complete_predicates() {
    let source = r#"select(op("a")) | union(op("b") or has_attr("tag")) | intersect(not op("c")) | except(attr("state", "skip"))"#;
    let parsed = parse(&lex(source));
    assert!(
        parsed.diagnostics().is_empty(),
        "{:?}",
        parsed.diagnostics()
    );
    assert!(matches!(
        parsed.program().unwrap().stages(),
        [
            Stage::Union {
                predicate: Predicate::Or { .. },
                ..
            },
            Stage::Intersect {
                predicate: Predicate::Not { .. },
                ..
            },
            Stage::Except {
                predicate: Predicate::Attr { .. },
                ..
            }
        ]
    ));
    assert_eq!(
        parsed.program().unwrap().stages()[0].range().as_range(),
        18..51
    );
}

#[test]
fn parser_builds_boolean_predicates_with_precedence_and_parentheses() {
    let parsed = parse(&lex(
        r#"select(op("a") or has_attr("tag") and not attr("state", "skip"))"#,
    ));
    let predicate = parsed.program().unwrap().predicate();
    assert!(matches!(
        predicate,
        Predicate::Or { predicates, .. }
            if matches!(predicates.as_slice(), [Predicate::Op { name, .. }, Predicate::And { predicates, .. }]
                if name == "a" && matches!(predicates.as_slice(), [Predicate::HasAttr { .. }, Predicate::Not { predicate, .. }]
                    if matches!(predicate.as_ref(), Predicate::Attr { name, value, .. } if name == "state" && value == "skip")))
    ));

    let grouped = parse(&lex(r#"select((op("a") or op("b")) and has_attr("tag"))"#));
    assert!(matches!(
        grouped.program().unwrap().predicate(),
        Predicate::And { predicates, .. }
            if matches!(predicates.first(), Some(Predicate::Group { predicate, range })
                if matches!(predicate.as_ref(), Predicate::Or { .. }) && range.as_range() == (7..27))
    ));
}

#[test]
fn malformed_and_over_nested_predicates_are_diagnosed() {
    let missing = parse(&lex(r#"select(attr("tag" "value"))"#));
    assert_eq!(missing.diagnostics()[0].message(), "expected `,` in attr");

    let at_limit = parse_with_nesting_limit(&lex(r#"select((op("x")))"#), 3);
    assert!(at_limit.diagnostics().is_empty());
    let beyond = parse_with_nesting_limit(&lex(r#"select(((op("x"))))"#), 3);
    assert_eq!(
        beyond.diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
}

#[test]
fn unary_and_boolean_chain_complexity_is_stack_safe() {
    let at_limit = format!("select({}op(\"x\"))", "not ".repeat(62));
    assert!(
        parse_with_nesting_limit(&lex(&at_limit), 64)
            .diagnostics()
            .is_empty()
    );
    let beyond = format!("select({}op(\"x\"))", "not ".repeat(63));
    assert_eq!(
        parse_with_nesting_limit(&lex(&beyond), 64).diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
    let pathological = format!("select({}op(\"x\"))", "not ".repeat(50_000));
    assert_eq!(
        parse(&lex(&pathological)).diagnostics()[0].message(),
        "query nesting limit exceeded"
    );

    for operator in [" or ", " and "] {
        let chain = format!(
            "select({})",
            std::iter::repeat_n("op(\"x\")", 10_000)
                .collect::<Vec<_>>()
                .join(operator)
        );
        let parsed = parse(&lex(&chain));
        match (operator, parsed.program().unwrap().predicate()) {
            (" or ", Predicate::Or { predicates, .. })
            | (" and ", Predicate::And { predicates, .. }) => {
                assert_eq!(predicates.len(), 10_000)
            }
            _ => panic!("expected a flat boolean chain"),
        }
    }
}

#[test]
fn parser_recovers_at_pipeline_boundaries_and_bounds_nesting() {
    let source = "select(op(\"x\")) | mystery(stuff) | count | root";
    let lexed = lex(source);
    let parsed = parse(&lexed);
    assert!(
        parsed
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message().contains("unknown"))
    );
    assert!(
        parsed
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message().contains("root requires"))
    );

    let valid = "select(op(\"x\"))";
    let lexed = lex(valid);
    let parsed = parse_with_nesting_limit(&lexed, 1);
    assert_eq!(
        parsed.diagnostics()[0].message(),
        "query nesting limit exceeded"
    );
    assert_eq!(parsed.diagnostics()[0].range().as_range(), 9..10);
}

#[test]
fn lexed_input_binds_the_parser_source_and_size_checks_do_not_wrap() {
    let source = String::from("select(op(\"é\"))");
    let lexed = lex(&source);
    let parsed = parse(&lexed);
    assert!(parsed.diagnostics().is_empty());
    assert!(
        matches!(parsed.program().unwrap().predicate(), Predicate::Op { name, .. } if name == "é")
    );

    assert!(query_size_supported(MAX_QUERY_BYTES));
    assert!(!query_size_supported(MAX_QUERY_BYTES.saturating_add(1)));
}

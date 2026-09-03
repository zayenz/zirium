use zirium::query::Query;

#[test]
fn parses_operation_name_selection_and_reports_positions() {
    let query = Query::parse("select(op(\"arith.addi\"))").unwrap();
    assert_eq!(query.operation_name(), "arith.addi");
    let error = Query::parse("select(op(arith.addi))").unwrap_err();
    assert!(error.position > 0);
    assert!(error.to_string().contains("expected"));

    let closure = Query::parse("select(op(\"arith.addi\")) | closure").unwrap();
    assert_eq!(closure.operation_name(), "arith.addi");
    Query::parse("select(op(\"arith.addi\")) | set_attr(\"analysis.tag\", \"hot\") | root")
        .unwrap();
    let error = Query::parse("select(op(\"arith.addi\")) | count | root").unwrap_err();
    assert!(error.to_string().contains("root requires a selection"));
    let error =
        Query::parse("select(op(\"arith.addi\")) | set_attr(\"bad name\", \"hot\")").unwrap_err();
    assert!(error.to_string().contains("dotted ASCII identifier"));
    let error =
        Query::parse("select(op(\"arith.addi\")) | set_attr(\"tag\", \"hot\nvalue\")").unwrap_err();
    assert!(error.to_string().contains("control characters"));
    Query::parse(r#"select(op("arith.addi")) | set_attr("tag", "quoted \"value\" \\ path")"#)
        .unwrap();
    assert!(Query::parse("select(op(\"arith.addi\")) | other").is_err());
}

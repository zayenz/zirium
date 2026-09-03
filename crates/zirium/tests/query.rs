use zirium::query::Query;

#[test]
fn parses_operation_name_selection_and_reports_positions() {
    let query = Query::parse("select(op(\"arith.addi\"))").unwrap();
    assert_eq!(query.operation_name(), "arith.addi");
    let error = Query::parse("select(op(arith.addi))").unwrap_err();
    assert!(error.position > 0);
    assert!(error.to_string().contains("expected"));
}

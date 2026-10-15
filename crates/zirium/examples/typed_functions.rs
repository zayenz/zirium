//! Native subtree cost and a destructured accumulator without generated query text.
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    query::{
        FunctionDefinition, FunctionLimits, FunctionNativeType as N, FunctionParameter,
        FunctionType as T, Located, PatternKind as P, PatternParameter, ReportExpr as E,
        ReportGenerator, ReportMatchArm, ValueBinaryOperator as B, op, ops, report,
    },
    semantic::{LoweringMode, lower_with_dialect_registry},
};
fn parameter(name: &str, declaration: T) -> FunctionParameter<N> {
    FunctionParameter {
        name: Located::new(name.to_owned(), 0..0),
        declaration,
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parsed = ParsedFile::parse_with_registry(
        include_bytes!("../../../examples/cli/subtree-cost.mlir").as_slice(),
        DialectRegistry::baseline(),
    )?;
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline())
            .document
            .unwrap();
    let subtree = FunctionDefinition::new(
        "subtree_cost",
        vec![parameter("op", T::Native(N::Operation))],
        T::Integer,
        vec![],
        E::cost_annotation(E::name("op"))
            .default(E::integer(1))
            .binary(
                B::Add,
                E::comprehension(
                    E::call("subtree_cost", [E::name("child")]),
                    [ReportGenerator::in_values(
                        "child",
                        E::direct_children(E::name("op")),
                        None,
                    )],
                )
                .sum(),
            ),
    )?;
    let root = ops().filter(op("example.root"));
    let cost = report(E::call("subtree_cost", [E::native(&root)])).with_functions(vec![subtree]);
    let mut budget = cost.new_budget(zirium::query::EvaluationLimits::default());
    let cost = cost.evaluate_output_with_budget(
        &document,
        DialectRegistry::baseline(),
        zirium::query::EvaluationOptions::default(),
        &mut budget,
    )?;
    println!("{}", cost.to_json_with_budget(&document, &mut budget)?);
    let n = E::name("n");
    let total = E::name("total");
    let name = |name: &str| Located::new(P::Name(Located::new(name.to_owned(), 0..0)), 0..0);
    let accumulate = FunctionDefinition::with_patterns(
        "accumulate",
        vec![PatternParameter {
            pattern: Located::new(P::Tuple(vec![name("n"), name("total")]), 0..0),
            declaration: T::Tuple(vec![T::Integer, T::Integer]),
        }],
        T::Integer,
        E::value_match(
            n.clone(),
            vec![
                ReportMatchArm {
                    pattern: Located::new(
                        P::Literal(zirium::query::ConstructedValue::Integer(0)),
                        0..0,
                    ),
                    guard: None,
                    value: total.clone(),
                },
                ReportMatchArm {
                    pattern: Located::new(P::Wildcard, 0..0),
                    guard: None,
                    value: E::block(
                        vec![(name("next_total"), total.binary(B::Add, n.clone()))],
                        E::call(
                            "accumulate",
                            [E::tuple([
                                n.binary(B::Subtract, E::integer(1)),
                                E::name("next_total"),
                            ])],
                        ),
                    )?,
                },
            ],
        )?,
    )?;
    let result = report(E::call(
        "accumulate",
        [E::tuple([E::integer(1000), E::integer(0)])],
    ))
    .with_functions(vec![accumulate])
    .with_function_limits(FunctionLimits { active_depth: 1 });
    let mut budget = result.new_budget(zirium::query::EvaluationLimits::default());
    let result = result.evaluate_output_with_budget(
        &document,
        DialectRegistry::baseline(),
        zirium::query::EvaluationOptions::default(),
        &mut budget,
    )?;
    println!("{}", result.to_json_with_budget(&document, &mut budget)?);
    Ok(())
}

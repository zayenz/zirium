//! Shared native reachability through typed builders, without generated text.
use zirium::{
    dialect::DialectRegistry,
    parser::ParsedFile,
    query::{
        EvaluationLimits, EvaluationOptions, FunctionDefinition, FunctionNativeType as N,
        FunctionType as T, Located, Pattern, PatternKind as P, PatternParameter, ReportExpr as E,
        ReportMatchArm, op, ops, report,
    },
    semantic::{LoweringMode, lower_with_dialect_registry},
};
fn pattern(kind: P) -> Pattern {
    Located::new(kind, 0..0)
}
fn name(value: &str) -> Pattern {
    pattern(P::Name(Located::new(value.to_owned(), 0..0)))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parsed = ParsedFile::parse_with_registry(
        include_bytes!("../../../examples/cli/worklist-reachability.mlir").as_slice(),
        DialectRegistry::baseline(),
    )?;
    let document =
        lower_with_dialect_registry(&parsed, LoweringMode::Strict, DialectRegistry::baseline())
            .document
            .unwrap();
    let expand = E::block(
        vec![(
            pattern(P::Record {
                fields: vec![(Located::new("targets".to_owned(), 0..0), name("targets"))],
                ignore_rest: true,
            }),
            E::user_info(E::name("current")),
        )],
        E::continue_([
            E::push_back_all(E::name("remaining"), E::name("targets")),
            E::insert(E::name("seen"), E::name("current")),
        ]),
    )?;
    let step = E::conditional(
        E::name("current").is_in(E::name("seen")),
        E::continue_([E::name("remaining"), E::name("seen")]),
        E::conditional(
            E::same_operation(E::name("current"), E::name("target")),
            E::break_(E::boolean(true)),
            expand,
        ),
    );
    let body = E::loop_(
        [
            ("pending".to_owned(), E::queue(E::array([E::name("start")]))),
            ("seen".to_owned(), E::identity_set(N::Operation)),
        ],
        E::value_match(
            E::pop_front(E::name("pending")),
            vec![
                ReportMatchArm {
                    pattern: pattern(P::Present(Box::new(pattern(P::Tuple(vec![
                        name("current"),
                        name("remaining"),
                    ]))))),
                    guard: None,
                    value: step,
                },
                ReportMatchArm {
                    pattern: pattern(P::Wildcard),
                    guard: None,
                    value: E::break_(E::boolean(false)),
                },
            ],
        )?,
    );
    let reaches = FunctionDefinition::with_patterns(
        "reaches",
        vec![PatternParameter {
            pattern: pattern(P::Tuple(vec![name("start"), name("target")])),
            declaration: T::Tuple(vec![T::Native(N::Operation), T::Native(N::Operation)]),
        }],
        T::Boolean,
        body,
    )?;
    let select = |name| {
        ops().filter(op(name)).evaluate(
            &document,
            DialectRegistry::baseline(),
            EvaluationLimits::default(),
        )
    };
    let start = select("graph.start")?[0];
    let target = select("graph.end")?[0];
    let disconnected = select("graph.disconnected")?[0];
    let call = |target| {
        E::call(
            "reaches",
            [E::tuple([E::operation(start), E::operation(target)])],
        )
    };
    let query = report(E::record([
        ("zero_edges", call(start)),
        ("reachable", call(target)),
        ("disconnected", call(disconnected)),
    ]))
    .with_functions(vec![reaches]);
    let mut budget = query.new_budget(EvaluationLimits::default());
    let output = query.evaluate_output_with_budget(
        &document,
        DialectRegistry::baseline(),
        EvaluationOptions::default(),
        &mut budget,
    )?;
    println!("{}", output.to_json_with_budget(&document, &mut budget)?);
    Ok(())
}

use std::{
    env, fs,
    io::{self, Read},
};

use zirium::{
    dialect::{DialectRegistry, RegistryConfig},
    parser::ParseDiagnosticKind,
    parser::ParsedFile,
    printer::PrintLayout,
    query::{EvaluationError, EvaluationLimits, Query, QueryOutput},
    semantic::{LoweringMode, RetentionProfile, lower_with_dialect_registry_and_retention},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("zirium: {error}");
        std::process::exit(1);
    }
}

const HELP: &str = r#"Usage: zirium [OPTIONS] [QUERY] [INPUT ...]
       zirium [OPTIONS] -f PROGRAM [INPUT ...]

Read MLIR from stdin when INPUT is omitted. An empty query prints the document.
Input files are independent; files are never overwritten. Options may appear
before or after QUERY. Use -- before paths beginning with a dash.

Options:
  -h, --help              Show this help
  --version               Show the version
  --preset NAME           Load a bundled dialect preset (repeatable)
  --list-presets          List bundled presets
  --registry FILE         Load a JSON registry (repeatable; combines with presets)
  -f, --program-file FILE Read the query from a file instead of an argument
  --strict                Reject incomplete parsing instead of warning
  --max-work N            Evaluation work limit (default 10000000)
  --max-items N           Maximum items per stream (default 1000000)

Examples:
  zirium 'filter(op("arith.addi")) | users | unique | count' input.mlir
  zirium --preset stablehlo --strict 'filter(op("stablehlo.dot_general")) | json' model.mlir
  zirium -f analysis.zirium model.mlir

Stages: input, filter(predicate), defs, defs(index), users, users(index), parent,
children, root(predicate), subtree, closure, slice, fixpoint(query), unique, attr("name"), names, result_types,
operand_types, set_attr("name", "value"), remove_attr("name"), emit, json, count.
Combine selections with union, intersect, except. Group them before counting.
Navigation preserves duplicates; use unique to count distinct operations.
Predicates: true, false, op("name"), dialect("name"), result_type("type"),
has_attr("name"), string_attr_eq("name", "value"); combine with not, and, or.
Reference: https://github.com/zayenz/zirium/blob/main/docs/query-language.md
"#;

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut registry_paths = Vec::new();
    let mut presets = Vec::new();
    let mut strict = false;
    let mut limits = EvaluationLimits::default();
    let mut program_path = None;
    let mut inline_query = None;
    let mut paths = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            "--version" => {
                println!("zirium {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--list-presets" => {
                for preset in DialectRegistry::preset_names() {
                    println!("{preset}");
                }
                return Ok(());
            }
            "--preset" => presets.push(arguments.next().ok_or("missing name after --preset")?),
            "--strict" => strict = true,
            "--max-work" | "--max-items" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("missing number after {argument}"))?
                    .parse::<usize>()
                    .map_err(|_| format!("{argument} requires a positive integer"))?;
                if value == 0 {
                    return Err(format!("{argument} requires a positive integer"));
                }
                if argument == "--max-work" {
                    limits.max_work = value;
                } else {
                    limits.max_items = value;
                }
            }
            "--registry" => {
                let path = arguments.next().ok_or("missing path after --registry")?;
                if path == "-" {
                    return Err(
                        "--registry requires a file path; stdin is reserved for MLIR".into(),
                    );
                }
                registry_paths.push(path);
            }
            "-f" | "--program-file" => {
                if program_path.is_some() || inline_query.is_some() {
                    return Err("supply one inline query or one program file".into());
                }
                program_path = Some(
                    arguments
                        .next()
                        .ok_or("missing program file after -f/--program-file")?,
                );
            }
            "--" => {
                if program_path.is_none() && inline_query.is_none() {
                    inline_query = arguments.next();
                }
                paths.extend(arguments);
                break;
            }
            option if option.starts_with('-') => return Err(format!("unknown option: {option}")),
            _ => {
                if program_path.is_some() || inline_query.is_some() {
                    paths.push(argument);
                } else {
                    inline_query = Some(argument);
                }
            }
        }
    }
    let query_text = if let Some(path) = program_path {
        fs::read_to_string(&path)
            .map_err(|error| format!("could not read program file {path}: {error}"))?
    } else {
        inline_query.unwrap_or_default()
    };
    let query = Query::parse(&query_text).map_err(|error| {
        let prefix = &query_text[..error.position];
        let line_number = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let line_start = prefix.rfind('\n').map_or(0, |offset| offset + 1);
        let line = query_text[line_start..].split('\n').next().unwrap_or("");
        let column = query_text[line_start..error.position].chars().count();
        format!(
            "{error} (line {line_number}, column {})\n{line}\n{}^",
            column + 1,
            " ".repeat(column)
        )
    })?;
    let registry = if registry_paths.is_empty() && presets.is_empty() {
        DialectRegistry::baseline().clone()
    } else {
        let mut configs = Vec::new();
        for path in registry_paths {
            let json = fs::read_to_string(&path)
                .map_err(|error| format!("could not load registry {path}: {error}"))?;
            configs.push(
                RegistryConfig::from_json(&json)
                    .map_err(|error| format!("could not load registry {path}: {error}"))?,
            );
        }
        if !presets.is_empty() {
            configs.push(RegistryConfig {
                presets,
                builtins: Vec::new(),
                operation_shapes: Vec::new(),
                operation_formats: Vec::new(),
            });
        }
        RegistryConfig::build_many(&configs)
            .map_err(|error| format!("could not load registry: {error}"))?
    };
    let registry = &registry;
    let inputs = if paths.is_empty() {
        vec![None]
    } else {
        paths.into_iter().map(Some).collect()
    };
    let mut answers = Vec::new();
    for path in inputs {
        let (name, bytes) = match path {
            Some(path) => {
                let bytes =
                    fs::read(&path).map_err(|error| format!("could not read {path}: {error}"))?;
                (path, bytes)
            }
            None => {
                let mut bytes = Vec::new();
                io::stdin()
                    .read_to_end(&mut bytes)
                    .map_err(|error| format!("could not read stdin: {error}"))?;
                ("stdin".to_owned(), bytes)
            }
        };
        let parsed = ParsedFile::parse_with_registry(bytes, registry)
            .map_err(|error| format!("could not parse {name}: {error}"))?;
        let recovered_unknown_custom =
            parsed.lexer_diagnostics().is_empty()
                && !parsed.syntax().diagnostics().is_empty()
                && parsed.syntax().diagnostics().iter().all(|diagnostic| {
                    diagnostic.kind() == ParseDiagnosticKind::UnknownCustomOperation
                });
        if !parsed.lexer_diagnostics().is_empty()
            || (!parsed.syntax().diagnostics().is_empty() && !recovered_unknown_custom)
        {
            let mut diagnostics = Vec::new();
            diagnostics.extend(parsed.lexer_diagnostics().iter().map(|diagnostic| {
                let range = diagnostic.range();
                format!(
                    "{:?} at bytes {}..{}",
                    diagnostic.kind(),
                    range.start(),
                    range.end()
                )
            }));
            diagnostics.extend(parsed.syntax().diagnostics().iter().map(|diagnostic| {
                let range = diagnostic.range();
                format!(
                    "{:?} at bytes {}..{}",
                    diagnostic.kind(),
                    range.start(),
                    range.end()
                )
            }));
            return Err(format!(
                "could not parse {name}: {}",
                diagnostics.join("; ")
            ));
        }
        if recovered_unknown_custom {
            let count = parsed.syntax().diagnostics().len();
            let first = parsed.syntax().diagnostics()[0].range();
            let message = format!(
                "{name}: incomplete semantic information ({count} recovered custom operations; first at bytes {}..{}); load a dialect with --preset NAME or --registry FILE",
                first.start(),
                first.end()
            );
            if strict {
                return Err(message);
            }
            eprintln!("zirium: warning: {message}; --strict rejects recovery");
        }
        let lowered = lower_with_dialect_registry_and_retention(
            &parsed,
            if recovered_unknown_custom {
                LoweringMode::BestEffort
            } else {
                LoweringMode::Strict
            },
            RetentionProfile::Hybrid,
            registry,
        );
        let mut document = lowered
            .document
            .filter(|_| lowered.diagnostics.is_empty())
            .ok_or_else(|| {
                let details = lowered
                    .diagnostics
                    .iter()
                    .enumerate()
                    .map(|(index, diagnostic)| {
                        let range = diagnostic.range;
                        format!(
                            "diagnostic #{} at bytes {}..{}: {}",
                            index + 1,
                            range.start(),
                            range.end(),
                            diagnostic.message
                        )
                    })
                    .collect::<Vec<_>>();
                let detail = if details.is_empty() {
                    "strict lowering failed".to_owned()
                } else {
                    details.join("; ")
                };
                format!("could not lower {name}: {detail}")
            })?;
        query
            .evaluate_with_limits(&mut document, registry, limits, |document, output| {
                let mut answer = Vec::new();
                let scalar = matches!(
                    output,
                    QueryOutput::Count(_)
                        | QueryOutput::Values(_)
                        | QueryOutput::Json(_)
                        | QueryOutput::Text(_)
                        | QueryOutput::Map(_)
                );
                match output {
                    QueryOutput::Operations(selected) => document
                        .write_selection(&mut answer, &selected, PrintLayout::Pretty, registry)
                        .map_err(|error| {
                            EvaluationError::new(format!("could not print {name}: {error}"))
                        })?,
                    QueryOutput::Count(count) => {
                        use std::io::Write;
                        writeln!(answer, "{count}")
                            .map_err(|error| EvaluationError::new(error.to_string()))?;
                    }
                    QueryOutput::Values(values) => {
                        use std::io::Write;
                        for value in values {
                            writeln!(answer, "{value}")
                                .map_err(|error| EvaluationError::new(error.to_string()))?;
                        }
                    }
                    QueryOutput::Map(values) => {
                        answer.extend_from_slice(
                            serde_json::to_string_pretty(&values)
                                .map_err(|error| EvaluationError::new(error.to_string()))?
                                .as_bytes(),
                        );
                        answer.push(b'\n');
                    }
                    QueryOutput::Json(json) | QueryOutput::Text(json) => {
                        answer.extend_from_slice(json.as_bytes())
                    }
                }
                answers.push((answer, scalar));
                Ok(())
            })
            .map_err(|error| format!("could not evaluate {name}: {error}"))?;
    }
    let stdout = io::stdout();
    let mut output = stdout.lock();
    use std::io::Write;
    let mut previous_scalar = true;
    for (index, (answer, scalar)) in answers.into_iter().enumerate() {
        if index != 0 && !scalar && !previous_scalar {
            output.write_all(b"// -----\n").map_err(|e| e.to_string())?;
        }
        output.write_all(&answer).map_err(|e| e.to_string())?;
        previous_scalar = scalar;
    }
    Ok(())
}

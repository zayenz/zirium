use std::{
    env, fs,
    io::{self, Read},
};

use zirium::{
    dialect::DialectRegistry,
    parser::ParseDiagnosticKind,
    parser::ParsedFile,
    printer::PrintLayout,
    query::{EvaluationError, Query, QueryOutput},
    semantic::{LoweringMode, RetentionProfile, lower_with_dialect_registry_and_retention},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("zirium: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut registry_paths = Vec::new();
    let mut program_path = None;
    let mut inline_query = None;
    let mut paths = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
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
                if program_path.is_some() {
                    return Err("program file may only be supplied once".into());
                }
                program_path = Some(
                    arguments
                        .next()
                        .ok_or("missing program file after -f/--program-file")?,
                );
            }
            "--" => {
                if program_path.is_none() {
                    inline_query = arguments.next();
                }
                paths.extend(arguments);
                break;
            }
            option if option.starts_with('-') => return Err(format!("unknown option: {option}")),
            _ => {
                if program_path.is_some() {
                    paths.push(argument);
                } else {
                    inline_query = Some(argument);
                }
                paths.extend(arguments);
                break;
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
    let registry = if registry_paths.is_empty() {
        DialectRegistry::proving().clone()
    } else {
        DialectRegistry::from_config_files(&registry_paths)
            .map_err(|error| format!("could not load registry: {error}"))?
    };
    let registry = &registry;
    let inputs = if paths.is_empty() {
        let mut bytes = Vec::new();
        io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|error| format!("could not read stdin: {error}"))?;
        vec![("stdin".to_owned(), bytes)]
    } else {
        paths
            .into_iter()
            .map(|path| {
                fs::read(&path)
                    .map(|bytes| (path.clone(), bytes))
                    .map_err(|error| format!("could not read {path}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut answers = Vec::new();
    for (name, bytes) in inputs {
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
            .evaluate(&mut document, registry, |document, output| {
                let mut answer = Vec::new();
                let scalar = matches!(output, QueryOutput::Count(_));
                match output {
                    QueryOutput::Selection(selected) => document
                        .write_selection(&mut answer, &selected, PrintLayout::Pretty, registry)
                        .map_err(|error| {
                            EvaluationError::new(format!("could not print {name}: {error}"))
                        })?,
                    QueryOutput::Count(count) => {
                        use std::io::Write;
                        writeln!(answer, "{count}")
                            .map_err(|error| EvaluationError::new(error.to_string()))?;
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

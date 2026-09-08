use std::{
    env, fs,
    io::{self, Read},
};

use zirium::{
    dialect::DialectRegistry,
    parser::ParseDiagnosticKind,
    parser::ParsedFile,
    printer::PrintLayout,
    query::{Query, QueryOutput},
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
            .trim()
            .to_owned()
    } else {
        inline_query.ok_or_else(|| "missing query; expected `select(op(\"name\"))`".to_owned())?
    };
    let query = Query::parse(&query_text).map_err(|error| error.to_string())?;
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
    let mut scalar_output = false;
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
        let result = query
            .evaluate(&mut document, registry)
            .map_err(|error| format!("could not evaluate {name}: {error}"))?;
        let mut answer = Vec::new();
        match result {
            QueryOutput::Selection(selected) => document
                .write_selection(&mut answer, &selected, PrintLayout::Pretty, registry)
                .map_err(|error| format!("could not print {name}: {error}"))?,
            QueryOutput::Root => {
                if !document.is_semantically_complete() {
                    return Err(format!(
                        "could not print {name}: cannot print an incomplete semantic document"
                    ));
                }
                document
                    .write_selection(
                        &mut answer,
                        document.root_operations(),
                        PrintLayout::Pretty,
                        registry,
                    )
                    .map_err(|error| format!("could not print {name}: {error}"))?
            }
            QueryOutput::Count(count) => {
                use std::io::Write;
                writeln!(answer, "{count}").map_err(|error| error.to_string())?;
                scalar_output = true;
            }
        }
        answers.push(answer);
    }
    let stdout = io::stdout();
    let mut output = stdout.lock();
    use std::io::Write;
    for (index, answer) in answers.into_iter().enumerate() {
        if index != 0 && !scalar_output {
            output.write_all(b"// -----\n").map_err(|e| e.to_string())?;
        }
        output.write_all(&answer).map_err(|e| e.to_string())?;
    }
    Ok(())
}

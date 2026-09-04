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
    let first_argument = arguments
        .next()
        .ok_or_else(|| "missing query; expected `select(op(\"name\"))`".to_owned())?;
    let query_text = if first_argument == "-f" || first_argument == "--program-file" {
        let path = arguments
            .next()
            .ok_or_else(|| format!("missing program file after `{first_argument}`"))?;
        let bytes = fs::read(&path)
            .map_err(|error| format!("could not read program file {path}: {error}"))?;
        String::from_utf8(bytes)
            .map_err(|_| format!("program file {path} is not valid UTF-8"))?
            .trim()
            .to_owned()
    } else {
        first_argument
    };
    let query = Query::parse(&query_text).map_err(|error| error.to_string())?;
    let paths = arguments.collect::<Vec<_>>();
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
    let registry = DialectRegistry::proving();
    let mut answers = Vec::new();
    let mut scalar_output = false;
    for (name, bytes) in inputs {
        let (bytes, insertion) = normalize_module_shorthand(bytes);
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
                let range = original_range(
                    diagnostic.range().start(),
                    diagnostic.range().end(),
                    insertion,
                );
                format!("{:?} at bytes {}..{}", diagnostic.kind(), range.0, range.1)
            }));
            diagnostics.extend(parsed.syntax().diagnostics().iter().map(|diagnostic| {
                let range = diagnostic.range();
                let range = original_range(range.start(), range.end(), insertion);
                format!("{:?} at bytes {}..{}", diagnostic.kind(), range.0, range.1)
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
        let mut document = lowered.document.ok_or_else(|| {
            let details = lowered
                .diagnostics
                .iter()
                .enumerate()
                .map(|(index, diagnostic)| {
                    let range =
                        original_range(diagnostic.range.start(), diagnostic.range.end(), insertion);
                    format!(
                        "diagnostic #{} at bytes {}..{}: {}",
                        index + 1,
                        range.0,
                        range.1,
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

fn normalize_module_shorthand(mut bytes: Vec<u8>) -> (Vec<u8>, Option<(usize, usize)>) {
    let mut position = 0;
    loop {
        while bytes.get(position).is_some_and(u8::is_ascii_whitespace) {
            position += 1;
        }
        if bytes.get(position..position + 2) == Some(b"//") {
            position = bytes[position..]
                .iter()
                .position(|&byte| byte == b'\n')
                .map_or(bytes.len(), |offset| position + offset + 1);
            continue;
        }
        break;
    }
    let end = position + b"module".len();
    if bytes.get(position..end) != Some(b"module") {
        return (bytes, None);
    }
    let mut brace = end;
    while bytes.get(brace).is_some_and(u8::is_ascii_whitespace) {
        brace += 1;
    }
    if bytes.get(brace) != Some(&b'{') {
        return (bytes, None);
    }
    const QUALIFIER: &[u8] = b"builtin.";
    bytes.splice(position..position, QUALIFIER.iter().copied());
    (bytes, Some((position, QUALIFIER.len())))
}

fn original_range(start: u32, end: u32, insertion: Option<(usize, usize)>) -> (u32, u32) {
    let Some((position, length)) = insertion else {
        return (start, end);
    };
    let map = |offset: u32| {
        if offset as usize <= position {
            offset
        } else {
            offset.saturating_sub(length as u32)
        }
    };
    (map(start), map(end))
}

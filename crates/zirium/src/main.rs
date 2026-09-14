use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use zirium::{
    dialect::{DialectRegistry, RegistryConfig},
    parser::ParseDiagnosticKind,
    parser::ParsedFile,
    printer::{FragmentScope, PrintLayout},
    query::{EvaluationError, EvaluationLimits, EvaluationOptions, Query, QueryOutput},
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

Read MLIR from stdin when INPUT is omitted or is -. An empty query prints the document.
Input files are independent; files are never overwritten. Options may appear
before or after QUERY. Use -- before paths beginning with a dash.

Options:
  -h, --help              Show this help
  --version               Show the version
  --preset NAME           Load a bundled dialect preset (repeatable)
  --list-presets          List bundled presets
  --registry FILE         Load a JSON registry (repeatable; combines with presets)
  -f, --program-file FILE Read the query from a file instead of an argument
  --strict                Reject incomplete parsing and unknown reachable references
  --jsonl                 Emit one attributable JSON record per line
  --ndjson                Alias for --jsonl
  --fragment-scope MODE   Selection shells: full (default) or minimal
  --max-work N            Evaluation work limit (default 10000000)
  --max-items N           Maximum items per stream (default 1000000)

Examples:
  zirium 'filter(op("arith.addi")) | users | unique | count' input.mlir
  zirium --preset stablehlo --strict 'filter(op("stablehlo.dot_general")) | json' model.mlir
  zirium -f analysis.zirium model.mlir

Stages: input, filter(predicate), defs, defs(index), users, users(index), parent,
children, root(predicate), subtree, closure, slice, reachable, fixpoint(query),
unique, attr("name"), names, result_types, operand_types, tally,
map_by(key, value), sort, sort_by(query), reverse, head(n), tail(n), min,
min_all, min_by(query), min_all_by(query), max, max_all, max_by(query),
max_all_by(query), set_attr("name", "value"),
remove_attr("name"), emit, json, markdown, print("text"), count.
Statements: prefix a query with `do` and end it with `;` to keep edits while
suppressing that statement's implicit result.
Combine selections with union, intersect, except. Group them before counting.
Navigation preserves duplicates; use unique to count distinct operations.
Predicates: true, false, op("name"), dialect("name"), result_type("type"),
has_attr("name"), string_attr_eq("name", "value"); combine with not, and, or.
Reference: https://github.com/zayenz/zirium/blob/main/docs/query-language.md
"#;

const OUTPUT_STAGING_MEMORY_LIMIT: usize = 1024 * 1024;

enum StagedStorage {
    Memory(Vec<u8>),
    File(TemporaryFile),
}

struct StagedOutput {
    storage: StagedStorage,
    memory_limit: usize,
    temporary_root: PathBuf,
}

impl StagedOutput {
    fn new(memory_limit: usize, temporary_root: PathBuf) -> Self {
        Self {
            storage: StagedStorage::Memory(Vec::new()),
            memory_limit,
            temporary_root,
        }
    }

    fn spill(&mut self) -> io::Result<()> {
        let StagedStorage::Memory(memory) = &self.storage else {
            return Ok(());
        };
        let mut temporary = TemporaryFile::create(&self.temporary_root)?;
        temporary.file_mut().write_all(memory)?;
        self.storage = StagedStorage::File(temporary);
        Ok(())
    }

    fn deliver_to(mut self, output: &mut impl Write) -> io::Result<()> {
        match &mut self.storage {
            StagedStorage::Memory(bytes) => output.write_all(bytes),
            StagedStorage::File(temporary) => {
                let file = temporary.file_mut();
                file.flush()?;
                file.seek(SeekFrom::Start(0))?;
                io::copy(file, output).map(|_| ())
            }
        }
    }

    fn deliver_stdout(self) -> Result<(), String> {
        let stdout = io::stdout();
        let mut output = stdout.lock();
        match self.deliver_to(&mut output) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }
}

impl Write for StagedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let StagedStorage::Memory(memory) = &mut self.storage
            && memory.len().saturating_add(bytes.len()) <= self.memory_limit
        {
            memory.extend_from_slice(bytes);
            return Ok(bytes.len());
        }
        self.spill()?;
        let StagedStorage::File(temporary) = &mut self.storage else {
            unreachable!("spilling replaces memory storage")
        };
        temporary.file_mut().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.storage {
            StagedStorage::Memory(_) => Ok(()),
            StagedStorage::File(temporary) => temporary.file_mut().flush(),
        }
    }
}

struct TemporaryFile {
    path: PathBuf,
    file: Option<File>,
}

impl TemporaryFile {
    fn create(root: &Path) -> io::Result<Self> {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

        for _ in 0..128 {
            let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                ".zirium-output-{}-{sequence}.tmp",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique output staging file",
        ))
    }

    fn file_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("temporary file remains open")
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let mut registry_paths = Vec::new();
    let mut presets = Vec::new();
    let mut strict = false;
    let mut ndjson = false;
    let mut fragment_scope = FragmentScope::Full;
    let mut limits = EvaluationLimits::default();
    let mut program_path = None;
    let mut inline_query = None;
    let mut paths = Vec::new();
    while let Some(argument) = arguments.next() {
        let option = argument.to_str().unwrap_or("");
        match option {
            "-h" | "--help" => {
                return write_stdout([HELP]);
            }
            "--version" => {
                return write_stdout([format!("zirium {}\n", env!("CARGO_PKG_VERSION"))]);
            }
            "--list-presets" => {
                return write_stdout(
                    DialectRegistry::preset_names()
                        .iter()
                        .map(|name| format!("{name}\n")),
                );
            }
            "--preset" => presets.push(
                arguments
                    .next()
                    .ok_or("missing name after --preset")?
                    .into_string()
                    .map_err(|_| "preset name must be UTF-8")?,
            ),
            "--strict" => strict = true,
            "--jsonl" | "--ndjson" => ndjson = true,
            "--fragment-scope" => {
                fragment_scope = match arguments
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .as_deref()
                {
                    Some("full") => FragmentScope::Full,
                    Some("minimal") => FragmentScope::Minimal,
                    Some(_) => return Err("--fragment-scope requires full or minimal".into()),
                    None => return Err("missing mode after --fragment-scope".into()),
                };
            }
            "--max-work" | "--max-items" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("missing number after {option}"))?
                    .to_str()
                    .and_then(|value| value.parse::<usize>().ok())
                    .ok_or_else(|| format!("{option} requires a positive integer"))?;
                if value == 0 {
                    return Err(format!("{option} requires a positive integer"));
                }
                if option == "--max-work" {
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
                    inline_query = arguments
                        .next()
                        .map(|query| query.into_string().map_err(|_| "query must be UTF-8"))
                        .transpose()?;
                }
                paths.extend(arguments);
                break;
            }
            option if option.starts_with('-') && option != "-" => {
                return Err(format!("unknown option: {option}"));
            }
            _ => {
                if program_path.is_some() || inline_query.is_some() {
                    paths.push(argument);
                } else {
                    inline_query = Some(argument.into_string().map_err(|_| "query must be UTF-8")?);
                }
            }
        }
    }
    let query_text = if let Some(path) = program_path {
        fs::read_to_string(&path).map_err(|error| {
            format!(
                "could not read program file {}: {error}",
                path.to_string_lossy()
            )
        })?
    } else {
        inline_query.unwrap_or_default()
    };
    let query = Query::parse_with_document_context(&query_text).map_err(|error| {
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
            let json = fs::read_to_string(&path).map_err(|error| {
                format!(
                    "could not load registry {}: {error}",
                    path.to_string_lossy()
                )
            })?;
            configs.push(RegistryConfig::from_json(&json).map_err(|error| {
                format!(
                    "could not load registry {}: {error}",
                    path.to_string_lossy()
                )
            })?);
        }
        if !presets.is_empty() {
            configs.push(RegistryConfig {
                presets,
                builtins: Vec::new(),
                operation_shapes: Vec::new(),
                operation_formats: Vec::new(),
                operation_alternatives: Vec::new(),
            });
        }
        RegistryConfig::build_many(&configs)
            .map_err(|error| format!("could not load registry: {error}"))?
    };
    let registry = &registry;
    let inputs = if paths.is_empty() {
        vec![None]
    } else {
        paths
            .into_iter()
            .map(|path| (path != "-").then_some(path))
            .collect()
    };
    let mut staged_output = StagedOutput::new(OUTPUT_STAGING_MEMORY_LIMIT, env::temp_dir());
    for path in inputs {
        let (name, bytes) = match path {
            Some(path) => {
                let bytes = fs::read(&path).map_err(|error| {
                    format!("could not read {}: {error}", path.to_string_lossy())
                })?;
                (path.to_string_lossy().into_owned(), bytes)
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
            .evaluate_with_context_options_and_limits(
                &mut document,
                registry,
                Some(&name),
                EvaluationOptions {
                    strict_unknown_references: strict,
                },
                limits,
                |document, output| {
                    if ndjson {
                        write_ndjson_record(
                            &mut staged_output,
                            document,
                            &name,
                            output,
                            registry,
                            fragment_scope,
                        )?;
                        return Ok(());
                    }
                    match output {
                        QueryOutput::Native(_) => {
                            return Err(EvaluationError::new(
                                "native query results require a library consumer",
                            ));
                        }
                        QueryOutput::Operations(selected) => document
                            .write_selection_with_scope(
                                &mut staged_output,
                                &selected,
                                PrintLayout::Pretty,
                                registry,
                                fragment_scope,
                            )
                            .map_err(|error| {
                                EvaluationError::new(format!("could not print {name}: {error}"))
                            })?,
                        QueryOutput::Count(count) => {
                            use std::io::Write;
                            writeln!(staged_output, "{count}").map_err(output_staging_error)?;
                        }
                        QueryOutput::Values(values) => {
                            use std::io::Write;
                            for value in values {
                                writeln!(staged_output, "{value}").map_err(output_staging_error)?;
                            }
                        }
                        QueryOutput::Array(values) => {
                            serde_json::to_writer_pretty(&mut staged_output, &values)
                                .map_err(|error| EvaluationError::new(error.to_string()))?;
                            staged_output
                                .write_all(b"\n")
                                .map_err(output_staging_error)?;
                        }
                        QueryOutput::Map(values) => {
                            serde_json::to_writer_pretty(&mut staged_output, &values)
                                .map_err(|error| EvaluationError::new(error.to_string()))?;
                            staged_output
                                .write_all(b"\n")
                                .map_err(output_staging_error)?;
                        }
                        QueryOutput::RankedMap(entries) => {
                            use serde::ser::{SerializeMap, Serializer};
                            let mut serializer = serde_json::Serializer::pretty(&mut staged_output);
                            let mut map = serializer
                                .serialize_map(Some(entries.len()))
                                .map_err(|error| EvaluationError::new(error.to_string()))?;
                            for (key, value) in entries {
                                map.serialize_entry(&key, &value)
                                    .map_err(|error| EvaluationError::new(error.to_string()))?;
                            }
                            map.end()
                                .map_err(|error| EvaluationError::new(error.to_string()))?;
                            staged_output
                                .write_all(b"\n")
                                .map_err(output_staging_error)?;
                        }
                        QueryOutput::Json(json) | QueryOutput::Text(json) => {
                            staged_output
                                .write_all(json.as_bytes())
                                .map_err(output_staging_error)?;
                        }
                    }
                    Ok(())
                },
            )
            .map_err(|error| format!("could not evaluate {name}: {error}"))?;
    }
    staged_output.deliver_stdout()
}

fn output_staging_error(error: io::Error) -> EvaluationError {
    EvaluationError::new(format!("could not stage output: {error}"))
}

fn write_ndjson_record(
    record: &mut impl Write,
    document: &zirium::semantic::Document,
    name: &str,
    output: QueryOutput,
    registry: &DialectRegistry,
    fragment_scope: FragmentScope,
) -> Result<(), EvaluationError> {
    use serde::ser::{SerializeMap, Serializer};

    record
        .write_all(b"{\"document\":")
        .map_err(output_staging_error)?;
    serde_json::to_writer(&mut *record, name)
        .map_err(|error| EvaluationError::new(error.to_string()))?;
    record
        .write_all(b",\"result\":")
        .map_err(output_staging_error)?;
    match output {
        QueryOutput::Json(json) => {
            // The emitter already produced valid JSON. Remove formatting outside
            // strings without reparsing objects into lexically ordered maps.
            let mut compact = Vec::with_capacity(8192);
            let mut in_string = false;
            let mut escaped = false;
            for byte in json.bytes() {
                if in_string {
                    compact.push(byte);
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        in_string = false;
                    }
                } else if !byte.is_ascii_whitespace() {
                    compact.push(byte);
                    in_string = byte == b'"';
                }
                if compact.len() == compact.capacity() {
                    record.write_all(&compact).map_err(output_staging_error)?;
                    compact.clear();
                }
            }
            record.write_all(&compact).map_err(output_staging_error)?;
        }
        QueryOutput::RankedMap(entries) => {
            let mut serializer = serde_json::Serializer::new(&mut *record);
            let mut map = serializer
                .serialize_map(Some(entries.len()))
                .map_err(|error| EvaluationError::new(error.to_string()))?;
            for (key, value) in entries {
                map.serialize_entry(&key, &value)
                    .map_err(|error| EvaluationError::new(error.to_string()))?;
            }
            map.end()
                .map_err(|error| EvaluationError::new(error.to_string()))?;
        }
        output => {
            let result = ndjson_value(document, output, registry, fragment_scope)?;
            serde_json::to_writer(&mut *record, &result)
                .map_err(|error| EvaluationError::new(error.to_string()))?;
        }
    }
    record.write_all(b"}\n").map_err(output_staging_error)?;
    Ok(())
}

fn ndjson_value(
    document: &zirium::semantic::Document,
    output: QueryOutput,
    registry: &DialectRegistry,
    fragment_scope: FragmentScope,
) -> Result<serde_json::Value, EvaluationError> {
    Ok(match output {
        QueryOutput::Native(_) => {
            return Err(EvaluationError::new(
                "native query results require a library consumer",
            ));
        }
        QueryOutput::Operations(selected) => {
            let mut bytes = Vec::new();
            document
                .write_selection_with_scope(
                    &mut bytes,
                    &selected,
                    PrintLayout::Pretty,
                    registry,
                    fragment_scope,
                )
                .map_err(|error| EvaluationError::new(error.to_string()))?;
            serde_json::Value::String(
                String::from_utf8(bytes)
                    .map_err(|error| EvaluationError::new(error.to_string()))?,
            )
        }
        QueryOutput::Values(values) => serde_json::json!(values),
        QueryOutput::Count(count) => serde_json::json!(count),
        QueryOutput::Map(values) => serde_json::Value::Object(values),
        QueryOutput::RankedMap(_) => unreachable!("ranked maps use ordered NDJSON serialization"),
        QueryOutput::Array(values) => serde_json::Value::Array(values),
        QueryOutput::Json(_) => unreachable!("JSON emissions retain their serialized order"),
        QueryOutput::Text(text) => serde_json::Value::String(text),
    })
}

fn write_stdout(chunks: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Result<(), String> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for chunk in chunks {
        if let Err(error) = output.write_all(chunk.as_ref()) {
            if error.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            }
            return Err(error.to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod staged_output_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create(name: &str) -> Self {
            static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "zirium-staged-output-test-{}-{name}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn entries(&self) -> usize {
            fs::read_dir(&self.0).unwrap().count()
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn spills_only_after_the_exact_memory_limit() {
        let directory = TestDirectory::create("boundary");
        let mut output = StagedOutput::new(4, directory.0.clone());

        output.write_all(b"abcd").unwrap();
        assert!(matches!(output.storage, StagedStorage::Memory(_)));
        assert_eq!(directory.entries(), 0);

        output.write_all(b"e").unwrap();
        assert!(matches!(output.storage, StagedStorage::File(_)));
        assert_eq!(directory.entries(), 1);

        let mut delivered = Vec::new();
        output.deliver_to(&mut delivered).unwrap();
        assert_eq!(delivered, b"abcde");
        assert_eq!(directory.entries(), 0);
    }

    #[test]
    fn spilled_chunks_keep_their_original_order_and_are_removed_on_drop() {
        let directory = TestDirectory::create("order");
        {
            let mut output = StagedOutput::new(3, directory.0.clone());
            for chunk in [b"ab".as_slice(), b"c", b"def", b"g"] {
                output.write_all(chunk).unwrap();
            }
            assert_eq!(directory.entries(), 1);

            let mut delivered = Vec::new();
            output.deliver_to(&mut delivered).unwrap();
            assert_eq!(delivered, b"abcdefg");
        }
        assert_eq!(directory.entries(), 0);

        {
            let mut output = StagedOutput::new(0, directory.0.clone());
            output.write_all(b"discarded").unwrap();
            assert_eq!(directory.entries(), 1);
        }
        assert_eq!(directory.entries(), 0);
    }

    #[test]
    fn spill_creation_failure_keeps_staged_bytes_private() {
        let directory = TestDirectory::create("failure");
        let missing_root = directory.0.join("missing");
        let mut output = StagedOutput::new(4, missing_root);
        output.write_all(b"abcd").unwrap();

        assert!(output.write_all(b"e").is_err());
        assert!(matches!(output.storage, StagedStorage::Memory(_)));
        assert_eq!(directory.entries(), 0);
    }
}

use std::{
    env,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use zirium::{
    dialect::{DialectRegistry, RegistryLoadOptions},
    diff::{ChangeField, ChangeKind, DiffLimits, DiffOptions, DiffSide, compare},
    parser::{ParseDiagnosticKind, ParseLimits, ParsedFile},
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
       zirium [OPTIONS] --diff BEFORE AFTER [QUERY]
       zirium [OPTIONS] --diff BEFORE AFTER -f PROGRAM

Read MLIR from stdin when INPUT is omitted or is -. An empty query prints the document.
Input files are independent; files are never overwritten. Options may appear
before or after QUERY. Use -- before paths beginning with a dash.
Value-taking long options accept separated or equals forms. `-f -` names a
literal query file named `-`; standard input may be an MLIR operand only once.

Options:
  -h, --help              Show this help
  --version               Show the version
  --preset NAME           Load a bundled dialect preset (repeatable)
  --list-presets          List bundled presets
  --registry FILE         Load a JSON registry (repeatable; combines with presets)
  --max-registry-depth N  Maximum registry import depth (default 64)
  --max-registry-files N  Maximum unique registry files (default 1024)
  --max-registry-edges N  Maximum declared registry imports (default 4096)
  --max-registry-bytes N  Maximum aggregate registry bytes (default 16777216)
  -f, --program-file FILE Read the query from a file instead of an argument
  --strict                Reject incomplete parsing and unknown reachable references
  --diff BEFORE AFTER     Compare two complete semantic MLIR documents
  --diff-locations        Include operation-attached locations in a diff
  --max-diff-work N       Semantic comparison work limit (default 10000000)
  --silent                Evaluate without writing query results to stdout
  --jsonl                 Emit one attributable JSON record per line
  --ndjson                Alias for --jsonl
  --fragment-scope MODE   Selection shells: full (default) or minimal
Parser limits:
  --max-file-bytes N      Maximum input size in bytes (default 4294967295)
Evaluator limits:
  --max-work N            Evaluation work limit (default 10000000)
  --max-items N           Maximum items per stream (default 1000000)

Examples:
  zirium 'filter(op("arith.addi")) | users | unique | count' input.mlir
  zirium --preset stablehlo --strict 'filter(op("stablehlo.dot_general")) | json' model.mlir
  zirium -f analysis.zirium model.mlir
  zirium --diff before.mlir after.mlir 'filter(changed("operands")) | json'

Stages: input, filter(predicate), defs, defs(index), users, users(index), parent,
children, root(predicate), subtree, closure, slice, reachable, fixpoint(query),
unique, attr("name"), names, result_types, operand_types, tally,
map_by(key, value), sort, sort_by(query), reverse, head(n), tail(n), min,
min_all, min_by(query), min_all_by(query), max, max_all, max_by(query),
max_all_by(query), set_attr("name", "value"),
remove_attr("name"), check, check(n), emit, json, markdown, print("text"), count.
Statements: prefix a query with `do` and end it with `;` to keep edits while
suppressing that statement's implicit result.
Combine selections with union, intersect, except. Group them before counting.
Navigation preserves duplicates; use unique to count distinct operations.
Predicates: true, false, op("name"), dialect("name"), result_type("type"),
has_attr("name"), string_attr_eq("name", "value"); combine with not, and, or.
Reference: https://github.com/zayenz/zirium/blob/main/docs/query-language.md
"#;

const OUTPUT_STAGING_MEMORY_LIMIT: usize = 1024 * 1024;
const DIAGNOSTIC_EXCERPT_COLUMNS: usize = 120;
const TAB_WIDTH: usize = 4;

fn source_diagnostic(
    source_name: &str,
    source: &[u8],
    range: Range<usize>,
    message: &str,
) -> String {
    let start = range.start.min(source.len());
    let end = range.end.max(start).min(source.len());
    let line_start = source[..start]
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |position| position + 1);
    let line_end = source[start..]
        .iter()
        .position(|&byte| byte == b'\n')
        .map_or(source.len(), |position| start + position);
    let line_end = if line_end > line_start && source[line_end - 1] == b'\r' {
        line_end - 1
    } else {
        line_end
    };
    let line_number = source[..line_start]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count()
        + 1;
    let before = diagnostic_text(&source[line_start..start.min(line_end)]);
    let selected = diagnostic_text(&source[start.min(line_end)..end.min(line_end)]);
    let column = display_width(&before) + 1;
    let marker_width = display_width(&selected).max(1);
    let rendered_line = expand_tabs(&diagnostic_text(&source[line_start..line_end]));
    let marker_offset = column - 1;
    let (excerpt, excerpt_offset, left_trimmed, right_trimmed) =
        bounded_excerpt(&rendered_line, marker_offset, DIAGNOSTIC_EXCERPT_COLUMNS);
    let marker_width = marker_width
        .min(DIAGNOSTIC_EXCERPT_COLUMNS.saturating_sub(excerpt_offset))
        .max(1);
    let left = if left_trimmed { "…" } else { "" };
    let right = if right_trimmed { "…" } else { "" };
    format!(
        "{source_name}:{line_number}:{column}: error: {message}\n{left}{excerpt}{right}\n{}^{}",
        " ".repeat(excerpt_offset + usize::from(left_trimmed)),
        "~".repeat(marker_width.saturating_sub(1))
    )
}

fn diagnostic_text(bytes: &[u8]) -> String {
    let mut text = String::new();
    let mut remaining = bytes;
    while !remaining.is_empty() {
        match std::str::from_utf8(remaining) {
            Ok(valid) => {
                text.push_str(valid);
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                text.push_str(std::str::from_utf8(&remaining[..valid]).unwrap());
                let invalid = error.error_len().unwrap_or(remaining.len() - valid);
                text.extend(std::iter::repeat_n('�', invalid));
                remaining = &remaining[valid + invalid..];
            }
        }
    }
    text
}

fn display_width(text: &str) -> usize {
    text.chars().fold(0, |column, character| {
        if character == '\t' {
            column + (TAB_WIDTH - column % TAB_WIDTH)
        } else {
            column + 1
        }
    })
}

fn expand_tabs(text: &str) -> String {
    let mut rendered = String::new();
    let mut column = 0;
    for character in text.chars() {
        if character == '\t' {
            let spaces = TAB_WIDTH - column % TAB_WIDTH;
            rendered.extend(std::iter::repeat_n(' ', spaces));
            column += spaces;
        } else {
            rendered.push(character);
            column += 1;
        }
    }
    rendered
}

fn bounded_excerpt(line: &str, marker: usize, limit: usize) -> (String, usize, bool, bool) {
    let characters = line.chars().collect::<Vec<_>>();
    if characters.len() <= limit {
        return (line.to_owned(), marker, false, false);
    }
    let start = marker
        .saturating_sub(limit / 2)
        .min(characters.len() - limit);
    let end = start + limit;
    (
        characters[start..end].iter().collect(),
        marker.saturating_sub(start),
        start > 0,
        end < characters.len(),
    )
}

fn lexer_diagnostic_message(kind: zirium::lexer::DiagnosticKind) -> &'static str {
    use zirium::lexer::DiagnosticKind;
    match kind {
        DiagnosticKind::FileLimit => "file size limit exceeded",
        DiagnosticKind::TokenLimit => "token limit exceeded",
        DiagnosticKind::InvalidByte => "invalid byte in input",
        DiagnosticKind::UnterminatedString => "unterminated string literal",
        DiagnosticKind::InvalidEscape => "invalid escape sequence",
        DiagnosticKind::InvalidIdentifier => "invalid identifier",
    }
}

fn parser_diagnostic_message(kind: ParseDiagnosticKind) -> String {
    match kind {
        ParseDiagnosticKind::Syntax => "invalid MLIR syntax".to_owned(),
        ParseDiagnosticKind::UnknownCustomOperation => "unknown custom operation".to_owned(),
        ParseDiagnosticKind::ShapeMismatch(shape) => {
            format!("custom operation does not match registered shape `{shape:?}`")
        }
        ParseDiagnosticKind::FormatMismatch => {
            "custom operation does not match its registered format".to_owned()
        }
        ParseDiagnosticKind::ProgressLimit => "parser recovery made no progress".to_owned(),
        ParseDiagnosticKind::DepthLimit => "delimiter nesting depth limit exceeded".to_owned(),
    }
}

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

fn read_bounded(mut input: impl Read, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    input
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > limit {
        return Err(format!("file size {} exceeds limit {limit}", bytes.len()));
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let mut registry_paths = Vec::new();
    let mut presets = Vec::new();
    let mut strict = false;
    let mut silent = false;
    let mut ndjson = false;
    let mut fragment_scope = FragmentScope::Full;
    let mut evaluation_limits = EvaluationLimits::default();
    let mut parse_limits = ParseLimits::default();
    let mut registry_limits = RegistryLoadOptions::default();
    let mut program_path = None;
    let mut inline_query = None;
    let mut paths = Vec::new();
    let mut diff_paths: Option<(OsString, OsString)> = None;
    let mut diff_options = DiffOptions::default();
    let mut diff_limits = DiffLimits::default();
    while let Some(argument) = arguments.next() {
        let option_text = argument.to_str().unwrap_or("");
        let (option, inline_value) = option_text
            .split_once('=')
            .filter(|(name, _)| {
                matches!(
                    *name,
                    "--preset"
                        | "--registry"
                        | "--program-file"
                        | "--fragment-scope"
                        | "--max-file-bytes"
                        | "--max-work"
                        | "--max-diff-work"
                        | "--max-items"
                        | "--max-registry-depth"
                        | "--max-registry-files"
                        | "--max-registry-edges"
                        | "--max-registry-bytes"
                )
            })
            .map_or((option_text, None), |(name, value)| (name, Some(value)));
        macro_rules! option_value {
            ($message:expr) => {
                inline_value
                    .map(OsString::from)
                    .or_else(|| arguments.next())
                    .ok_or($message)?
            };
        }
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
                option_value!("missing name after --preset")
                    .into_string()
                    .map_err(|_| "preset name must be UTF-8")?,
            ),
            "--strict" => strict = true,
            "--diff" => {
                if diff_paths.is_some() {
                    return Err("--diff may be supplied only once".into());
                }
                let before = arguments.next().ok_or("missing BEFORE path after --diff")?;
                let after = arguments.next().ok_or_else(|| {
                    format!(
                        "missing AFTER path after --diff (BEFORE was {})",
                        before.to_string_lossy()
                    )
                })?;
                diff_paths = Some((before, after));
            }
            "--diff-locations" => diff_options.compare_locations = true,
            "--silent" => silent = true,
            "--jsonl" | "--ndjson" => ndjson = true,
            "--fragment-scope" => {
                fragment_scope = match inline_value
                    .map(OsString::from)
                    .or_else(|| arguments.next())
                    .and_then(|value| value.into_string().ok())
                    .as_deref()
                {
                    Some("full") => FragmentScope::Full,
                    Some("minimal") => FragmentScope::Minimal,
                    Some(_) => return Err("--fragment-scope requires full or minimal".into()),
                    None => return Err("missing mode after --fragment-scope".into()),
                };
            }
            "--max-file-bytes"
            | "--max-work"
            | "--max-diff-work"
            | "--max-items"
            | "--max-registry-depth"
            | "--max-registry-files"
            | "--max-registry-edges"
            | "--max-registry-bytes" => {
                let requirement =
                    if option == "--max-file-bytes" || option.starts_with("--max-registry-") {
                        "a non-negative integer"
                    } else {
                        "a positive integer"
                    };
                let value = option_value!(format!("missing number after {option}"))
                    .to_str()
                    .and_then(|value| value.parse::<usize>().ok())
                    .ok_or_else(|| format!("{option} requires {requirement}"))?;
                if value == 0 && matches!(option, "--max-work" | "--max-diff-work" | "--max-items")
                {
                    return Err(format!("{option} requires a positive integer"));
                }
                match option {
                    "--max-file-bytes" => parse_limits.max_file_bytes = value,
                    "--max-work" => evaluation_limits.max_work = value,
                    "--max-diff-work" => diff_limits.max_work = value,
                    "--max-items" => evaluation_limits.max_items = value,
                    "--max-registry-depth" => registry_limits.max_depth = value,
                    "--max-registry-files" => registry_limits.max_files = value,
                    "--max-registry-edges" => registry_limits.max_edges = value,
                    _ => registry_limits.max_bytes = value,
                }
            }
            "--registry" => {
                let path = option_value!("missing path after --registry");
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
                program_path = Some(option_value!(
                    "missing program file after -f/--program-file"
                ));
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
    let (query_name, query_text) = if let Some(path) = program_path {
        if diff_paths.is_some() && path == "-" {
            return Err(
                "-f - is unavailable in diff mode because stdin is reserved for MLIR".into(),
            );
        }
        let text = fs::read_to_string(&path).map_err(|error| {
            format!(
                "could not read program file {}: {error}",
                path.to_string_lossy()
            )
        })?;
        (path.to_string_lossy().into_owned(), text)
    } else {
        ("<query>".to_owned(), inline_query.unwrap_or_default())
    };
    let registry = if registry_paths.is_empty() && presets.is_empty() {
        DialectRegistry::baseline().clone()
    } else {
        DialectRegistry::from_config_files_with_options_and_presets(
            registry_paths,
            presets,
            registry_limits,
        )
        .map_err(|error| format!("could not load registry: {error}"))?
    };
    let registry = &registry;
    if diff_paths.is_none() && diff_options.compare_locations {
        return Err("--diff-locations requires --diff".into());
    }
    if diff_paths.is_none() && diff_limits.max_work != DiffLimits::default().max_work {
        return Err("--max-diff-work requires --diff".into());
    }
    if let Some((before, after)) = diff_paths {
        if !paths.is_empty() {
            return Err("diff mode accepts only BEFORE, AFTER, and one optional query".into());
        }
        diff_limits.max_changes = evaluation_limits.max_items;
        return run_diff(
            before,
            after,
            &query_name,
            &query_text,
            registry,
            parse_limits,
            diff_options,
            diff_limits,
            fragment_scope,
            ndjson,
            silent,
        );
    }
    let query = Query::parse_with_document_context(&query_text).map_err(|error| {
        let message = format!("query error: {}", error.message);
        source_diagnostic(
            &query_name,
            query_text.as_bytes(),
            error.position..error.position,
            &message,
        )
    })?;
    let inputs = if paths.is_empty() {
        vec![None]
    } else {
        if paths.iter().filter(|path| path.as_os_str() == "-").count() > 1 {
            return Err("standard input may be specified only once".into());
        }
        paths
            .into_iter()
            .map(|path| (path != "-").then_some(path))
            .collect()
    };
    let mut staged_output = StagedOutput::new(OUTPUT_STAGING_MEMORY_LIMIT, env::temp_dir());
    for path in inputs {
        let (name, bytes) = match path {
            Some(path) => {
                let file = File::open(&path).map_err(|error| {
                    format!("could not read {}: {error}", path.to_string_lossy())
                })?;
                let bytes = read_bounded(file, parse_limits.max_file_bytes).map_err(|error| {
                    format!("could not read {}: {error}", path.to_string_lossy())
                })?;
                (path.to_string_lossy().into_owned(), bytes)
            }
            None => {
                let bytes = read_bounded(io::stdin().lock(), parse_limits.max_file_bytes)
                    .map_err(|error| format!("could not read stdin: {error}"))?;
                ("stdin".to_owned(), bytes)
            }
        };
        let parsed = ParsedFile::parse_with_limits_and_registry(bytes, parse_limits, registry)
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
                source_diagnostic(
                    &name,
                    parsed.original_bytes(),
                    range.start() as usize..range.end() as usize,
                    lexer_diagnostic_message(diagnostic.kind()),
                )
            }));
            diagnostics.extend(parsed.syntax().diagnostics().iter().map(|diagnostic| {
                let range = diagnostic.range();
                source_diagnostic(
                    &name,
                    parsed.original_bytes(),
                    range.start() as usize..range.end() as usize,
                    &parser_diagnostic_message(diagnostic.kind()),
                )
            }));
            return Err(format!(
                "could not parse {name}:\n{}",
                diagnostics.join("\n")
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
                    .map(|diagnostic| {
                        let range = diagnostic.range;
                        source_diagnostic(
                            &name,
                            parsed.original_bytes(),
                            range.start() as usize..range.end() as usize,
                            &diagnostic.message,
                        )
                    })
                    .collect::<Vec<_>>();
                let detail = if details.is_empty() {
                    "strict lowering failed".to_owned()
                } else {
                    details.join("\n")
                };
                format!("could not lower {name}:\n{detail}")
            })?;
        drop(parsed);
        query
            .evaluate_with_context_options_and_limits(
                &mut document,
                registry,
                Some(&name),
                EvaluationOptions {
                    strict_unknown_references: strict,
                },
                evaluation_limits,
                |document, output| {
                    if silent {
                        return Ok(());
                    }
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

#[allow(clippy::too_many_arguments)]
fn run_diff(
    before_path: OsString,
    after_path: OsString,
    query_name: &str,
    query_text: &str,
    registry: &DialectRegistry,
    parse_limits: ParseLimits,
    options: DiffOptions,
    limits: DiffLimits,
    fragment_scope: FragmentScope,
    ndjson: bool,
    silent: bool,
) -> Result<(), String> {
    if before_path == "-" && after_path == "-" {
        return Err("both diff inputs cannot read from stdin".into());
    }
    if query_text.contains("set_attr(") || query_text.contains("remove_attr(") {
        return Err(format!(
            "could not evaluate {query_name}: diff queries are read-only"
        ));
    }
    let (before_name, before) = load_diff_input(before_path, registry, parse_limits, "before")?;
    let (after_name, after) = load_diff_input(after_path, registry, parse_limits, "after")?;
    let diff = compare(&before, &after, registry, options, limits)
        .map_err(|error| format!("could not compare {before_name} and {after_name}: {error}"))?;
    let mut output = StagedOutput::new(OUTPUT_STAGING_MEMORY_LIMIT, env::temp_dir());
    if !silent {
        evaluate_diff_cli(
            &diff,
            query_text,
            &before_name,
            &after_name,
            fragment_scope,
            ndjson,
            &mut output,
        )?;
    }
    output.deliver_stdout()
}

fn load_diff_input(
    path: OsString,
    registry: &DialectRegistry,
    limits: ParseLimits,
    side: &str,
) -> Result<(String, zirium::semantic::Document), String> {
    let (name, bytes) = if path == "-" {
        (
            "stdin".to_owned(),
            read_bounded(io::stdin().lock(), limits.max_file_bytes)
                .map_err(|error| format!("could not read {side} input stdin: {error}"))?,
        )
    } else {
        let name = path.to_string_lossy().into_owned();
        let file = File::open(&path)
            .map_err(|error| format!("could not read {side} input {name}: {error}"))?;
        let bytes = read_bounded(file, limits.max_file_bytes)
            .map_err(|error| format!("could not read {side} input {name}: {error}"))?;
        (name, bytes)
    };
    let parsed = ParsedFile::parse_with_limits_and_registry(bytes, limits, registry)
        .map_err(|error| format!("could not parse {side} input {name}: {error}"))?;
    if let Some(diagnostic) = parsed
        .lexer_diagnostics()
        .first()
        .map(|diagnostic| {
            let range = diagnostic.range();
            source_diagnostic(
                &name,
                parsed.original_bytes(),
                range.start() as usize..range.end() as usize,
                lexer_diagnostic_message(diagnostic.kind()),
            )
        })
        .or_else(|| {
            parsed.syntax().diagnostics().first().map(|diagnostic| {
                let range = diagnostic.range();
                source_diagnostic(
                    &name,
                    parsed.original_bytes(),
                    range.start() as usize..range.end() as usize,
                    &parser_diagnostic_message(diagnostic.kind()),
                )
            })
        })
    {
        return Err(format!(
            "could not parse {side} input {name}:\n{diagnostic}"
        ));
    }
    let lowered = lower_with_dialect_registry_and_retention(
        &parsed,
        LoweringMode::Strict,
        RetentionProfile::SemanticOnly,
        registry,
    );
    let document = lowered.document.ok_or_else(|| {
        let details = lowered
            .diagnostics
            .iter()
            .map(|diagnostic| {
                source_diagnostic(
                    &name,
                    parsed.original_bytes(),
                    diagnostic.range.start() as usize..diagnostic.range.end() as usize,
                    &diagnostic.message,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("could not lower {side} input {name}:\n{details}")
    })?;
    Ok((name, document))
}

enum DiffCliValue {
    Changes(Vec<zirium::diff::ChangeId>),
    Operations(DiffSide, Vec<zirium::semantic::OperationId>),
    Count(usize),
    Names(Vec<String>),
    Json(String),
}

fn evaluate_diff_cli(
    diff: &zirium::diff::Diff<'_>,
    source: &str,
    before_name: &str,
    after_name: &str,
    fragment_scope: FragmentScope,
    ndjson: bool,
    output: &mut StagedOutput,
) -> Result<(), String> {
    if source.trim().is_empty() {
        if ndjson {
            let result: serde_json::Value =
                serde_json::from_str(&diff.to_json().map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            serde_json::to_writer(
                &mut *output,
                &serde_json::json!({
                    "schema": "zirium.diff.v1",
                    "before_document": before_name,
                    "after_document": after_name,
                    "result_side": serde_json::Value::Null,
                    "comparison": {
                        "locations": if diff.options().compare_locations { "compare" } else { "ignore" },
                        "opaque_values": "bytes",
                        "opaque_before": diff.statistics().opaque_before,
                        "opaque_after": diff.statistics().opaque_after,
                        "ambiguous_groups": diff.statistics().ambiguous_groups,
                        "bounded_fallback_groups": diff.statistics().bounded_fallback_groups,
                    },
                    "result": result,
                }),
            ).map_err(|e| e.to_string())?;
            output.write_all(b"\n").map_err(|e| e.to_string())?;
        } else {
            output
                .write_all(format!("{before_name} -> {after_name}\n{}", diff.to_text()).as_bytes())
                .map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    if source.contains(';') || source.contains('=') {
        return Err(
            "diff query statements and bindings are not yet supported by this build".into(),
        );
    }
    let stages = split_diff_pipeline(source)?;
    let mut value = DiffCliValue::Changes(diff.change_ids().collect());
    for stage in stages {
        let stage = stage.trim();
        value = if stage == "input" {
            DiffCliValue::Changes(diff.change_ids().collect())
        } else if stage.starts_with("filter(") {
            filter_diff_value(diff, value, stage)?
        } else if stage == "before" || stage == "after" {
            let DiffCliValue::Changes(changes) = value else {
                return Err(format!("`{stage}` requires a change stream"));
            };
            let side = if stage == "before" {
                DiffSide::Before
            } else {
                DiffSide::After
            };
            let operations = changes
                .into_iter()
                .filter_map(|id| diff.endpoint_id(id, side).ok().flatten())
                .collect();
            DiffCliValue::Operations(side, operations)
        } else if stage == "unique" {
            match value {
                DiffCliValue::Changes(items) => DiffCliValue::Changes(unique(items)),
                DiffCliValue::Operations(side, items) => {
                    DiffCliValue::Operations(side, unique(items))
                }
                _ => return Err("unique requires a change or operation stream".into()),
            }
        } else if stage == "count" {
            DiffCliValue::Count(match value {
                DiffCliValue::Changes(items) => items.len(),
                DiffCliValue::Operations(_, items) => items.len(),
                DiffCliValue::Names(items) => items.len(),
                _ => return Err("count requires a stream".into()),
            })
        } else if stage == "names" {
            match value {
                DiffCliValue::Changes(items) => DiffCliValue::Names(
                    items
                        .into_iter()
                        .filter_map(|id| {
                            let (side, op) = diff.representative_id(id).ok()?;
                            diff.document(side).operation_name(op).map(str::to_owned)
                        })
                        .collect(),
                ),
                DiffCliValue::Operations(side, items) => DiffCliValue::Names(
                    items
                        .into_iter()
                        .filter_map(|op| diff.document(side).operation_name(op).map(str::to_owned))
                        .collect(),
                ),
                _ => return Err("names requires a change or operation stream".into()),
            }
        } else if stage == "json" {
            DiffCliValue::Json(match value {
                DiffCliValue::Changes(items) => {
                    diff.selection_to_json(&items).map_err(|e| e.to_string())?
                }
                DiffCliValue::Operations(side, items) => {
                    operation_json(diff.document(side), &items)
                }
                DiffCliValue::Names(items) => {
                    serde_json::to_string_pretty(&items).map_err(|e| e.to_string())?
                }
                DiffCliValue::Count(count) => serde_json::to_string(&count).unwrap(),
                DiffCliValue::Json(_) => return Err("json cannot be applied twice".into()),
            })
        } else {
            return Err(format!("unknown or unsupported diff query stage `{stage}`"));
        };
    }
    match value {
        DiffCliValue::Changes(items) => {
            let json = diff.selection_to_json(&items).map_err(|e| e.to_string())?;
            if ndjson {
                write_diff_envelope(
                    output,
                    before_name,
                    after_name,
                    None,
                    serde_json::from_str(&json).unwrap(),
                )?;
            } else {
                output
                    .write_all(json.as_bytes())
                    .map_err(|e| e.to_string())?;
                output.write_all(b"\n").map_err(|e| e.to_string())?;
            }
        }
        DiffCliValue::Operations(side, items) => {
            if ndjson {
                write_diff_envelope(
                    output,
                    before_name,
                    after_name,
                    Some(side),
                    serde_json::Value::String(selection_text(
                        diff.document(side),
                        &items,
                        diff.registry(),
                        fragment_scope,
                    )?),
                )?;
            } else {
                diff.document(side)
                    .write_selection_with_scope(
                        output,
                        &items,
                        PrintLayout::Pretty,
                        diff.registry(),
                        fragment_scope,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        DiffCliValue::Count(count) => writeln!(output, "{count}").map_err(|e| e.to_string())?,
        DiffCliValue::Names(items) => {
            for item in items {
                writeln!(output, "{item}").map_err(|e| e.to_string())?;
            }
        }
        DiffCliValue::Json(json) => {
            output
                .write_all(json.as_bytes())
                .map_err(|e| e.to_string())?;
            output.write_all(b"\n").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn split_diff_pipeline(source: &str) -> Result<Vec<&str>, String> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut start = 0usize;
    let mut result = Vec::new();
    for (index, byte) in source.bytes().enumerate() {
        match byte {
            b'"' => quoted = !quoted,
            b'(' if !quoted => depth += 1,
            b')' if !quoted => depth = depth.checked_sub(1).ok_or("unmatched `)` in diff query")?,
            b'|' if !quoted && depth == 0 => {
                result.push(&source[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if quoted || depth != 0 {
        return Err("unterminated string or group in diff query".into());
    }
    result.push(&source[start..]);
    Ok(result)
}

fn filter_diff_value(
    diff: &zirium::diff::Diff<'_>,
    value: DiffCliValue,
    stage: &str,
) -> Result<DiffCliValue, String> {
    let DiffCliValue::Changes(items) = value else {
        return Err("change predicates require a change stream".into());
    };
    let calls = ["change", "changed", "op", "dialect"]
        .into_iter()
        .flat_map(|name| {
            extract_string_calls(stage, name)
                .into_iter()
                .map(move |value| (name, value))
        })
        .collect::<Vec<_>>();
    if calls.is_empty() {
        return Err("unsupported diff filter predicate".into());
    }
    let mut result = Vec::new();
    'item: for id in items {
        let change = diff.change(id).map_err(|e| e.to_string())?;
        for (name, argument) in &calls {
            let matches = match *name {
                "change" => {
                    let kind: ChangeKind = argument
                        .parse()
                        .map_err(|e: zirium::diff::DiffError| e.to_string())?;
                    if kind == ChangeKind::Moved {
                        change.moved()
                    } else {
                        change.kind() == kind
                    }
                }
                "changed" => {
                    let field: ChangeField = argument
                        .parse()
                        .map_err(|e: zirium::diff::DiffError| e.to_string())?;
                    change.fields().contains(&field)
                }
                "op" | "dialect" => {
                    let (side, op) = diff.representative_id(id).map_err(|e| e.to_string())?;
                    let op_name = diff.document(side).operation_name(op).unwrap_or("");
                    if *name == "op" {
                        op_name == argument
                    } else {
                        op_name
                            .split_once('.')
                            .map_or(op_name, |(dialect, _)| dialect)
                            == argument
                    }
                }
                _ => unreachable!(),
            };
            if !matches {
                continue 'item;
            }
        }
        result.push(id);
    }
    Ok(DiffCliValue::Changes(result))
}

fn extract_string_calls(source: &str, name: &str) -> Vec<String> {
    let prefix = format!("{name}(\"");
    let mut rest = source;
    let mut values = Vec::new();
    while let Some(start) = rest.find(&prefix) {
        rest = &rest[start + prefix.len()..];
        let Some(end) = rest.find("\")") else {
            break;
        };
        values.push(rest[..end].to_owned());
        rest = &rest[end + 2..];
    }
    values
}

fn unique<T: Copy + Eq + std::hash::Hash>(items: Vec<T>) -> Vec<T> {
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|item| seen.insert(*item))
        .collect()
}

fn operation_json(
    document: &zirium::semantic::Document,
    operations: &[zirium::semantic::OperationId],
) -> String {
    let values = operations.iter().map(|operation| serde_json::json!({
        "name": document.operation_name(*operation),
        "range": document.operation_source_range(*operation).map(|range| [range.start(), range.end()]),
    })).collect::<Vec<_>>();
    serde_json::to_string_pretty(&values).unwrap()
}

fn selection_text(
    document: &zirium::semantic::Document,
    operations: &[zirium::semantic::OperationId],
    registry: &DialectRegistry,
    scope: FragmentScope,
) -> Result<String, String> {
    let mut bytes = Vec::new();
    document
        .write_selection_with_scope(&mut bytes, operations, PrintLayout::Pretty, registry, scope)
        .map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

fn write_diff_envelope(
    output: &mut StagedOutput,
    before: &str,
    after: &str,
    side: Option<DiffSide>,
    result: serde_json::Value,
) -> Result<(), String> {
    serde_json::to_writer(&mut *output, &serde_json::json!({
        "schema": "zirium.diff.v1", "before_document": before, "after_document": after,
        "result_side": side.map(|side| if side == DiffSide::Before { "before" } else { "after" }), "result": result,
    })).map_err(|e| e.to_string())?;
    output.write_all(b"\n").map_err(|e| e.to_string())
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
        QueryOutput::Operations(selected) => {
            record.write_all(b"\"").map_err(output_staging_error)?;
            {
                let mut buffered = BufWriter::with_capacity(8192, &mut *record);
                let mut writer = JsonStringWriter::new(&mut buffered);
                document
                    .write_selection_with_scope(
                        &mut writer,
                        &selected,
                        PrintLayout::Pretty,
                        registry,
                        fragment_scope,
                    )
                    .map_err(|error| EvaluationError::new(error.to_string()))?;
                writer.flush().map_err(output_staging_error)?;
            }
            record.write_all(b"\"").map_err(output_staging_error)?;
        }
        output => {
            let result = ndjson_value(output)?;
            serde_json::to_writer(&mut *record, &result)
                .map_err(|error| EvaluationError::new(error.to_string()))?;
        }
    }
    record.write_all(b"}\n").map_err(output_staging_error)?;
    Ok(())
}

struct JsonStringWriter<W> {
    output: W,
}

impl<W> JsonStringWriter<W> {
    fn new(output: W) -> Self {
        Self { output }
    }
}

impl<W: Write> Write for JsonStringWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut unchanged = 0;
        for (index, &byte) in bytes.iter().enumerate() {
            let escaped: &[u8] = match byte {
                b'\"' => br#"\""#,
                b'\\' => br#"\\"#,
                b'\x08' => br#"\b"#,
                b'\t' => br#"\t"#,
                b'\n' => br#"\n"#,
                b'\x0c' => br#"\f"#,
                b'\r' => br#"\r"#,
                0x00..=0x1f => {
                    self.output.write_all(&bytes[unchanged..index])?;
                    self.output.write_all(&[
                        b'\\',
                        b'u',
                        b'0',
                        b'0',
                        HEX[(byte >> 4) as usize],
                        HEX[(byte & 0xf) as usize],
                    ])?;
                    unchanged = index + 1;
                    continue;
                }
                _ => continue,
            };
            self.output.write_all(&bytes[unchanged..index])?;
            self.output.write_all(escaped)?;
            unchanged = index + 1;
        }
        self.output.write_all(&bytes[unchanged..])?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

fn ndjson_value(output: QueryOutput) -> Result<serde_json::Value, EvaluationError> {
    Ok(match output {
        QueryOutput::Native(_) => {
            return Err(EvaluationError::new(
                "native query results require a library consumer",
            ));
        }
        QueryOutput::Operations(_) => unreachable!("operation selections stream into NDJSON"),
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

    #[test]
    fn json_string_writer_matches_serde_json_escaping() {
        let value = "\0\x01\x08\t\n\x0c\r\x1f\"\\é";
        let expected = serde_json::to_vec(value).unwrap();
        let mut actual = Vec::new();
        JsonStringWriter::new(&mut actual)
            .write_all(value.as_bytes())
            .unwrap();
        assert_eq!(actual, expected[1..expected.len() - 1]);
    }

    struct GrowingFile {
        reader: File,
        writer: File,
        initial_bytes: usize,
        grew: bool,
    }

    impl Read for GrowingFile {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let available = if self.grew {
                buffer.len()
            } else {
                buffer.len().min(self.initial_bytes)
            };
            let read = self.reader.read(&mut buffer[..available])?;
            if !self.grew {
                self.writer.write_all(b"e")?;
                self.writer.flush()?;
                self.grew = true;
            }
            Ok(read)
        }
    }

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
    fn bounded_read_rejects_a_file_that_grows_after_the_initial_read() {
        let directory = TestDirectory::create("growing-input");
        let path = directory.0.join("input.mlir");
        fs::write(&path, b"abcd").unwrap();
        let input = GrowingFile {
            reader: File::open(&path).unwrap(),
            writer: OpenOptions::new().append(true).open(&path).unwrap(),
            initial_bytes: 4,
            grew: false,
        };

        assert_eq!(
            read_bounded(input, 4).unwrap_err(),
            "file size 5 exceeds limit 4"
        );
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

    #[test]
    fn source_diagnostics_handle_unicode_invalid_utf8_crlf_tabs_and_eof() {
        let unicode = source_diagnostic("input.mlir", "\téx".as_bytes(), 3..4, "bad token");
        assert!(unicode.starts_with("input.mlir:1:6: error: bad token\n    éx\n     ^"));

        let invalid = source_diagnostic("bytes.mlir", b"ok\xffx", 2..3, "invalid byte");
        assert_eq!(invalid, "bytes.mlir:1:3: error: invalid byte\nok�x\n  ^");

        let crlf = source_diagnostic("input.mlir", b"first\r\nsecond\r\n", 7..8, "bad");
        assert_eq!(crlf, "input.mlir:2:1: error: bad\nsecond\n^");

        let eof = source_diagnostic("input.mlir", b"last\n", 5..5, "unexpected EOF");
        assert_eq!(eof, "input.mlir:2:1: error: unexpected EOF\n\n^");
    }

    #[test]
    fn source_diagnostics_bound_long_lines_around_the_marker() {
        let source = format!("{}BAD{}", "a".repeat(200), "z".repeat(200));
        let diagnostic = source_diagnostic("long.mlir", source.as_bytes(), 200..203, "bad");
        let mut lines = diagnostic.lines();
        assert_eq!(lines.next(), Some("long.mlir:1:201: error: bad"));
        let excerpt = lines.next().unwrap();
        assert!(excerpt.starts_with('…') && excerpt.ends_with('…'));
        assert!(excerpt.chars().count() <= DIAGNOSTIC_EXCERPT_COLUMNS + 2);
        assert!(lines.next().unwrap().contains("^~~"));
    }
}

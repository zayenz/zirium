//! Source-aware layout formatting for textual MLIR.

use std::{
    collections::{HashMap, HashSet},
    fmt, io,
};

use crate::{
    SyntaxKind, SyntaxTree,
    dialect::DialectRegistry,
    lexer::{Token, TokenKind},
    parser::{ParseFileError, ParsedFile},
    printer::{DialectPrintMode, FragmentScope, PreserveError, PrintError, PrintLayout},
    semantic::{Document, OperationId},
    source::TextRange,
};

/// Chooses where operation assembly spellings come from before layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AssemblyStyle {
    /// Retain the custom or generic spelling represented by the source CST.
    #[default]
    Source,
    /// Regenerate operations in MLIR's quoted generic assembly form.
    Generic,
}

/// Options for source-aware MLIR formatting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatOptions {
    pub assembly: AssemblyStyle,
    pub line_width: usize,
    pub indent_width: usize,
}

/// Largest indentation width accepted by formatting entry points.
pub const MAX_INDENT_WIDTH: usize = 256;

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            assembly: AssemblyStyle::Source,
            line_width: 100,
            indent_width: 2,
        }
    }
}

/// Failure while formatting parsed syntax.
#[derive(Debug)]
pub enum FormatError {
    MissingSyntax,
    GenericAssemblyRequiresSemantics,
    InvalidIndentWidth(usize),
    Parse(ParseFileError),
    Print(PrintError),
    Preserve(PreserveError),
    Io(io::Error),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSyntax => f.write_str("formatted output requires retained syntax"),
            Self::GenericAssemblyRequiresSemantics => {
                f.write_str("generic assembly formatting requires a semantic document")
            }
            Self::InvalidIndentWidth(width) => write!(
                f,
                "format indent must be between 1 and {MAX_INDENT_WIDTH}, got {width}"
            ),
            Self::Parse(error) => error.fmt(f),
            Self::Print(error) => error.fmt(f),
            Self::Preserve(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse(error) => Some(error),
            Self::Print(error) => Some(error),
            Self::Preserve(error) => Some(error),
            Self::MissingSyntax
            | Self::GenericAssemblyRequiresSemantics
            | Self::InvalidIndentWidth(_) => None,
        }
    }
}

impl From<io::Error> for FormatError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ParseFileError> for FormatError {
    fn from(error: ParseFileError) -> Self {
        Self::Parse(error)
    }
}

impl From<PrintError> for FormatError {
    fn from(error: PrintError) -> Self {
        Self::Print(error)
    }
}

impl From<PreserveError> for FormatError {
    fn from(error: PreserveError) -> Self {
        Self::Preserve(error)
    }
}

impl ParsedFile {
    /// Formats the retained assembly spelling while normalizing layout.
    ///
    /// This method does not lower the file. Operation names, SSA names, aliases,
    /// comments, and custom assembly forms therefore remain source-derived.
    pub fn write_formatted<W: io::Write>(
        &self,
        sink: &mut W,
        options: FormatOptions,
    ) -> Result<(), FormatError> {
        validate_options(options)?;
        if options.assembly == AssemblyStyle::Generic {
            return Err(FormatError::GenericAssemblyRequiresSemantics);
        }
        format_syntax(sink, self.original_bytes(), self.syntax().tree(), options)
    }

    /// Returns source-aware formatted output as bytes.
    pub fn formatted_bytes(&self, options: FormatOptions) -> Result<Vec<u8>, FormatError> {
        let mut output = Vec::with_capacity(self.original_bytes().len());
        self.write_formatted(&mut output, options)?;
        Ok(output)
    }
}

impl Document {
    /// Formats a complete semantic document, retaining source assembly when it
    /// is still available and falling back to registered custom printers.
    pub fn write_formatted<W: io::Write>(
        &self,
        sink: &mut W,
        registry: &DialectRegistry,
        options: FormatOptions,
    ) -> Result<(), FormatError> {
        self.write_formatted_selection(
            sink,
            &self.operations().collect::<Vec<_>>(),
            registry,
            FragmentScope::Full,
            options,
        )
    }

    /// Formats an operation selection with the structural shells needed to
    /// retain its source location.
    pub fn write_formatted_selection<W: io::Write>(
        &self,
        sink: &mut W,
        selected: &[OperationId],
        registry: &DialectRegistry,
        scope: FragmentScope,
        options: FormatOptions,
    ) -> Result<(), FormatError> {
        validate_options(options)?;
        let all = self.operations().collect::<Vec<_>>();
        if options.assembly == AssemblyStyle::Source && selected == all {
            if self.dirty_operations().is_empty()
                && self.dirty_blocks().is_empty()
                && let (Some(source), Some(tree)) = (self.source_bytes(), self.syntax_tree())
            {
                return format_syntax(sink, source, tree, options);
            }
            if self.source_bytes().is_some() && self.syntax_tree().is_some() {
                let generated =
                    self.preserving_bytes_with_registry(PrintLayout::Compact, registry)?;
                let parsed = ParsedFile::parse_with_registry(generated, registry)?;
                return parsed.write_formatted(
                    sink,
                    FormatOptions {
                        assembly: AssemblyStyle::Source,
                        ..options
                    },
                );
            }
        }

        let mode = match options.assembly {
            AssemblyStyle::Source => DialectPrintMode::PreferCustom,
            AssemblyStyle::Generic => DialectPrintMode::GenericOnly,
        };
        let mut generated = Vec::new();
        self.write_selection_with_scope_and_mode(
            &mut generated,
            selected,
            PrintLayout::Compact,
            registry,
            scope,
            mode,
        )?;
        let parsed = ParsedFile::parse_with_registry(generated, registry)?;
        parsed.write_formatted(
            sink,
            FormatOptions {
                assembly: AssemblyStyle::Source,
                ..options
            },
        )
    }

    pub fn formatted_bytes(
        &self,
        registry: &DialectRegistry,
        options: FormatOptions,
    ) -> Result<Vec<u8>, FormatError> {
        let mut output = Vec::new();
        self.write_formatted(&mut output, registry, options)?;
        Ok(output)
    }
}

fn validate_options(options: FormatOptions) -> Result<(), FormatError> {
    if !(1..=MAX_INDENT_WIDTH).contains(&options.indent_width) {
        return Err(FormatError::InvalidIndentWidth(options.indent_width));
    }
    Ok(())
}

pub(crate) fn format_syntax<W: io::Write>(
    sink: &mut W,
    source: &[u8],
    tree: &SyntaxTree,
    options: FormatOptions,
) -> Result<(), FormatError> {
    let annotations = Annotations::new(tree);
    let tokens = tree.tokens(tree.root()).ok_or(FormatError::MissingSyntax)?;
    let mut writer = LayoutWriter::new(sink, options);
    let mut index = 0usize;
    let mut pending_blank = false;
    let mut had_whitespace = false;
    let mut had_newline = false;

    while index < tokens.len() {
        let token = tokens[index];
        if token.kind() == TokenKind::Eof {
            break;
        }
        if token.kind() == TokenKind::Whitespace {
            let bytes = slice(source, token.range());
            pending_blank |= bytes.iter().filter(|&&byte| byte == b'\n').count() >= 2;
            had_whitespace = true;
            had_newline |= bytes.contains(&b'\n');
            index += 1;
            continue;
        }
        if let Some(verbatim) = annotations.verbatim.get(&token.range().start()).copied() {
            if let Some(&indent) = annotations.line_starts.get(&token.range().start()) {
                writer.start_construct(indent, pending_blank)?;
            } else {
                writer.prepare_space(had_whitespace);
            }
            let bytes = slice(source, verbatim.range);
            match verbatim.kind {
                VerbatimKind::Block => writer.write_verbatim_block(bytes)?,
                VerbatimKind::Atom => writer.write_atom(bytes)?,
            }
            while index < tokens.len() && tokens[index].range().end() <= verbatim.range.end() {
                index += 1;
            }
            pending_blank = false;
            had_whitespace = false;
            had_newline = false;
            continue;
        }
        if let Some(&indent) = annotations.line_starts.get(&token.range().start()) {
            writer.start_construct(indent, pending_blank)?;
        }
        if token.kind() == TokenKind::LineComment {
            if had_newline {
                writer.start_construct(writer.indent, pending_blank)?;
            }
            writer.write_comment(slice(source, token.range()), had_whitespace && !had_newline)?;
            pending_blank = false;
            had_whitespace = false;
            had_newline = false;
            index += 1;
            continue;
        }
        pending_blank = false;
        let is_region_open = annotations
            .region_opens
            .contains_key(&token.range().start());
        let is_region_close = annotations
            .region_closes
            .contains_key(&token.range().start());
        let space_before = annotations.space_before.contains(&token.range().start());
        let comparison_prefix = matches!(token.kind(), TokenKind::Less | TokenKind::Greater)
            && tokens
                .get(index + 1)
                .is_some_and(|next| next.kind() == TokenKind::Equal);
        if is_region_close {
            let indent = annotations.region_closes[&token.range().start()];
            writer.start_construct(indent, false)?;
        }
        writer.write_token(
            slice(source, token.range()),
            token.kind(),
            had_whitespace,
            is_region_open,
            space_before,
            comparison_prefix,
        )?;
        if is_region_open {
            let indent = annotations.region_opens[&token.range().start()];
            writer.hard_line(indent + 1)?;
        }
        had_whitespace = false;
        had_newline = false;
        index += 1;
    }
    writer.finish()?;
    Ok(())
}

fn slice(source: &[u8], range: TextRange) -> &[u8] {
    &source[range.start() as usize..range.end() as usize]
}

#[derive(Clone, Copy)]
enum VerbatimKind {
    Atom,
    Block,
}

#[derive(Clone, Copy)]
struct VerbatimRange {
    range: TextRange,
    kind: VerbatimKind,
}

struct Annotations {
    line_starts: HashMap<u32, usize>,
    region_opens: HashMap<u32, usize>,
    region_closes: HashMap<u32, usize>,
    verbatim: HashMap<u32, VerbatimRange>,
    space_before: HashSet<u32>,
}

impl Annotations {
    fn new(tree: &SyntaxTree) -> Self {
        let mut annotations = Self {
            line_starts: HashMap::new(),
            region_opens: HashMap::new(),
            region_closes: HashMap::new(),
            verbatim: HashMap::new(),
            space_before: HashSet::new(),
        };
        let Some(nodes) = tree.subtree(tree.root()) else {
            return annotations;
        };
        for node in nodes {
            let Some(kind) = tree.kind(node) else {
                continue;
            };
            let Some(tokens) = tree.tokens(node) else {
                continue;
            };
            let Some(first) = tokens.iter().find(|token| !is_trivia(token.kind())) else {
                continue;
            };
            let indent = syntax_indent(tree, node);
            match kind {
                SyntaxKind::Operation
                | SyntaxKind::DialectOperation
                | SyntaxKind::UnparsedCustomOperation
                | SyntaxKind::AliasDefinition => {
                    annotations
                        .line_starts
                        .insert(first.range().start(), indent);
                    if matches!(kind, SyntaxKind::Operation | SyntaxKind::DialectOperation)
                        && let Some(elements) = tree.elements(node)
                    {
                        for element in elements {
                            if let crate::SyntaxElement::Token { token, .. } = element
                                && token.kind() == TokenKind::Colon
                            {
                                annotations.space_before.insert(token.range().start());
                            }
                        }
                    }
                }
                SyntaxKind::Block if first.kind() == TokenKind::CaretIdentifier => {
                    annotations
                        .line_starts
                        .insert(first.range().start(), indent);
                }
                SyntaxKind::Region => {
                    let open = tokens
                        .iter()
                        .find(|token| token.kind() == TokenKind::LBrace);
                    let close = tokens
                        .iter()
                        .rev()
                        .find(|token| token.kind() == TokenKind::RBrace);
                    if let Some(open) = open {
                        annotations
                            .region_opens
                            .insert(open.range().start(), indent);
                    }
                    if let Some(close) = close {
                        annotations
                            .region_closes
                            .insert(close.range().start(), indent);
                        annotations
                            .line_starts
                            .insert(close.range().start(), indent);
                    }
                }
                _ => {}
            }
            let verbatim_kind = match kind {
                SyntaxKind::UnparsedCustomOperation | SyntaxKind::Error => {
                    Some(VerbatimKind::Block)
                }
                SyntaxKind::OpaqueAttribute
                | SyntaxKind::OpaqueType
                | SyntaxKind::DenseElementsAttribute
                | SyntaxKind::SparseElementsAttribute
                | SyntaxKind::DenseResourceElementsAttribute => Some(VerbatimKind::Atom),
                _ => None,
            };
            if let Some(verbatim_kind) = verbatim_kind
                && !has_verbatim_ancestor(tree, node)
                && let Some(range) = significant_range(tokens)
            {
                annotations.verbatim.insert(
                    range.start(),
                    VerbatimRange {
                        range,
                        kind: verbatim_kind,
                    },
                );
            }
        }
        annotations
    }
}

fn significant_range(tokens: &[Token]) -> Option<TextRange> {
    let first = tokens.iter().find(|token| !is_trivia(token.kind()))?;
    let last = tokens.iter().rev().find(|token| !is_trivia(token.kind()))?;
    TextRange::new(first.range().start(), last.range().end())
}

fn has_verbatim_ancestor(tree: &SyntaxTree, node: crate::NodeId) -> bool {
    let mut parent = tree.parent(node);
    while let Some(current) = parent {
        if matches!(
            tree.kind(current),
            Some(
                SyntaxKind::UnparsedCustomOperation
                    | SyntaxKind::Error
                    | SyntaxKind::OpaqueAttribute
                    | SyntaxKind::OpaqueType
                    | SyntaxKind::DenseElementsAttribute
                    | SyntaxKind::SparseElementsAttribute
                    | SyntaxKind::DenseResourceElementsAttribute
            )
        ) {
            return true;
        }
        parent = tree.parent(current);
    }
    false
}

fn syntax_indent(tree: &SyntaxTree, node: crate::NodeId) -> usize {
    let mut indent = 0usize;
    let mut parent = tree.parent(node);
    while let Some(current) = parent {
        match tree.kind(current) {
            Some(SyntaxKind::Region) => indent += 1,
            Some(SyntaxKind::Block)
                if tree.tokens(current).is_some_and(|tokens| {
                    tokens
                        .iter()
                        .find(|token| !is_trivia(token.kind()))
                        .is_some_and(|token| token.kind() == TokenKind::CaretIdentifier)
                }) =>
            {
                indent += 1;
            }
            _ => {}
        }
        parent = tree.parent(current);
    }
    indent
}

fn is_trivia(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Whitespace | TokenKind::LineComment | TokenKind::Eof
    )
}

struct LayoutWriter<'a, W> {
    sink: &'a mut W,
    options: FormatOptions,
    column: usize,
    line_start: bool,
    indent: usize,
    base_indent: usize,
    pending_space: bool,
    comma_break: bool,
    previous: Option<TokenKind>,
}

impl<'a, W: io::Write> LayoutWriter<'a, W> {
    fn new(sink: &'a mut W, options: FormatOptions) -> Self {
        Self {
            sink,
            options,
            column: 0,
            line_start: true,
            indent: 0,
            base_indent: 0,
            pending_space: false,
            comma_break: false,
            previous: None,
        }
    }

    fn start_construct(&mut self, indent: usize, blank: bool) -> io::Result<()> {
        if !self.line_start {
            self.newline()?;
        }
        if blank && self.column == 0 {
            self.sink.write_all(b"\n")?;
        }
        self.indent = indent;
        self.base_indent = indent;
        self.pending_space = false;
        self.comma_break = false;
        self.previous = None;
        Ok(())
    }

    fn hard_line(&mut self, indent: usize) -> io::Result<()> {
        if !self.line_start {
            self.newline()?;
        }
        self.indent = indent;
        self.base_indent = indent;
        self.pending_space = false;
        self.comma_break = false;
        self.previous = None;
        Ok(())
    }

    fn prepare_space(&mut self, source_had_space: bool) {
        self.pending_space |= source_had_space;
    }

    fn write_comment(&mut self, bytes: &[u8], source_had_space: bool) -> io::Result<()> {
        if !self.line_start && source_had_space {
            self.write_space()?;
        }
        self.write_bytes(bytes)?;
        self.newline()?;
        self.previous = None;
        Ok(())
    }

    fn write_token(
        &mut self,
        bytes: &[u8],
        kind: TokenKind,
        source_had_space: bool,
        region_open: bool,
        force_space_before: bool,
        comparison_prefix: bool,
    ) -> io::Result<()> {
        let previous = self.previous;
        let composite_equal = kind == TokenKind::Equal
            && matches!(
                previous,
                Some(TokenKind::Equal | TokenKind::Less | TokenKind::Greater)
            )
            && !source_had_space;
        let result_number = kind == TokenKind::HashIdentifier
            && previous == Some(TokenKind::PercentIdentifier)
            && !source_had_space;
        let force_before = force_space_before
            || (kind == TokenKind::Equal && !composite_equal)
            || kind == TokenKind::Arrow
            || region_open
            || comparison_prefix
            || binary_operator(kind);
        let suppress_before = closes_delimiter(kind)
            || kind == TokenKind::Comma
            || (kind == TokenKind::Greater && !comparison_prefix)
            || (kind == TokenKind::Colon && !force_space_before)
            || previous.is_some_and(opens_delimiter)
            || previous == Some(TokenKind::Less)
            || kind == TokenKind::X
            || previous == Some(TokenKind::X)
            || composite_equal
            || result_number;
        let lexical_space = previous.is_some_and(word_like) && word_like(kind);
        let wants_space = !suppress_before
            && (force_before || self.pending_space || source_had_space || lexical_space);
        if self.comma_break
            && self.options.line_width > 0
            && self.column + usize::from(wants_space) + bytes.len() > self.options.line_width
        {
            self.newline()?;
            self.indent = self.base_indent + 1;
        } else if wants_space {
            self.write_space()?;
        }
        self.write_bytes(bytes)?;
        self.pending_space = matches!(kind, TokenKind::Comma | TokenKind::Equal | TokenKind::Arrow)
            || (kind == TokenKind::Colon && force_space_before)
            || binary_operator(kind);
        self.comma_break = kind == TokenKind::Comma;
        self.previous = Some(kind);
        Ok(())
    }

    fn write_atom(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.pending_space {
            self.write_space()?;
        }
        self.write_bytes(bytes)?;
        self.pending_space = false;
        self.comma_break = false;
        self.previous = Some(TokenKind::BareIdentifier);
        Ok(())
    }

    fn write_verbatim_block(&mut self, bytes: &[u8]) -> io::Result<()> {
        let text = trim_ascii_whitespace(bytes);
        for (line_index, line) in text.split(|&byte| byte == b'\n').enumerate() {
            if line_index != 0 {
                self.newline()?;
            }
            self.write_bytes(if line_index == 0 {
                line
            } else {
                trim_horizontal_whitespace(line)
            })?;
        }
        self.pending_space = false;
        self.comma_break = false;
        self.previous = Some(TokenKind::BareIdentifier);
        Ok(())
    }

    fn write_space(&mut self) -> io::Result<()> {
        if !self.line_start && self.column > 0 {
            self.sink.write_all(b" ")?;
            self.column += 1;
        }
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.line_start {
            let spaces = self.indent.saturating_mul(self.options.indent_width);
            for _ in 0..spaces {
                self.sink.write_all(b" ")?;
            }
            self.column = spaces;
            self.line_start = false;
        }
        self.sink.write_all(bytes)?;
        if let Some(last_newline) = bytes.iter().rposition(|&byte| byte == b'\n') {
            self.column = bytes.len() - last_newline - 1;
            self.line_start = bytes.ends_with(b"\n");
        } else {
            self.column += bytes.len();
        }
        Ok(())
    }

    fn newline(&mut self) -> io::Result<()> {
        self.sink.write_all(b"\n")?;
        self.column = 0;
        self.line_start = true;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        if !self.line_start {
            self.newline()?;
        }
        Ok(())
    }
}

fn opens_delimiter(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::LParen | TokenKind::LBracket | TokenKind::Less
    )
}

fn closes_delimiter(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace
    )
}

fn binary_operator(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Plus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::VerticalBar
            | TokenKind::FloorDiv
            | TokenKind::CeilDiv
            | TokenKind::Mod
    )
}

fn word_like(kind: TokenKind) -> bool {
    !matches!(
        kind,
        TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::Less
            | TokenKind::Greater
            | TokenKind::Colon
            | TokenKind::Comma
            | TokenKind::Equal
            | TokenKind::Plus
            | TokenKind::Star
            | TokenKind::VerticalBar
            | TokenKind::Minus
            | TokenKind::Slash
            | TokenKind::Arrow
            | TokenKind::Question
            | TokenKind::X
    )
}

fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn trim_horizontal_whitespace(mut bytes: &[u8]) -> &[u8] {
    while bytes
        .first()
        .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
    {
        bytes = &bytes[1..];
    }
    bytes
}

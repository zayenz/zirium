//! Owning parse results, resource limits, text edits, and typed CST views.
//!
//! Parsing is lossless and diagnostic-driven. A successful [`ParsedFile`] may
//! still contain lexer or syntax diagnostics, so consumers inspect both before
//! assuming that the input is well formed. Fatal errors are reserved for source
//! size and compact-tree construction failures.

use crate::{
    dialect::{DialectRegistry, OperationDescriptor, OperationShape},
    lexer::{Diagnostic as LexDiagnostic, Lexed, LexerLimits, TokenKind, lex_with_limits},
    representation::{
        CompactError, CompletedMarker, EventBuilder, Marker, NodeId, SyntaxKind, SyntaxTree,
    },
    source::{Source, SourceError, TextRange},
};

/// One replacement in the original source byte coordinate space.
///
/// Edit ranges may be empty for insertion. [`ParsedFile::apply_text_edits`]
/// sorts edits by range before applying them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEdit {
    pub range: TextRange,
    pub replacement: std::sync::Arc<[u8]>,
}

/// A range-level error in a batch of source edits.
#[derive(Debug, PartialEq, Eq)]
pub enum TextEditError {
    OutOfBounds(TextRange),
    Overlapping(TextRange),
}

impl std::fmt::Display for TextEditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds(range) => write!(f, "text edit range {range} is out of bounds"),
            Self::Overlapping(range) => {
                write!(f, "text edit range {range} overlaps another edit")
            }
        }
    }
}

impl std::error::Error for TextEditError {}

/// An immutable, owning parse result.
///
/// This value keeps the input bytes, lexer diagnostics, parser diagnostics,
/// and compact CST together. Syntax errors do not make construction fail.
#[derive(Debug)]
pub struct ParsedFile {
    source: Source,
    lexer_diagnostics: Vec<LexDiagnostic>,
    syntax: ParsedSyntax,
    limits: ParseLimits,
}

/// Resource limits shared by parsing and later semantic lowering.
///
/// Lexer limits take effect during [`ParsedFile`] construction. Attribute and
/// alias depth limits take effect when the file is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseLimits {
    pub max_file_bytes: usize,
    pub max_tokens: usize,
    pub max_delimiter_depth: usize,
    pub max_payload_bytes: usize,
    pub max_numeric_literal_bytes: usize,
    pub max_attribute_depth: usize,
    pub max_alias_expansion_depth: usize,
}

impl Default for ParseLimits {
    fn default() -> Self {
        let lexer = LexerLimits::default();
        let parser = ParserLimits::default();
        Self {
            max_file_bytes: lexer.max_file_bytes,
            max_tokens: lexer.max_tokens,
            max_delimiter_depth: parser.max_delimiter_depth,
            max_payload_bytes: parser.max_payload_bytes,
            max_numeric_literal_bytes: parser.max_numeric_literal_bytes,
            max_attribute_depth: 64,
            max_alias_expansion_depth: 64,
        }
    }
}

impl ParsedFile {
    /// Parses and owns bytes using default limits and generic operation syntax.
    ///
    /// The result remains available for malformed MLIR; inspect
    /// [`Self::lexer_diagnostics`] and [`ParsedSyntax::diagnostics`].
    ///
    /// # Errors
    ///
    /// Returns an error when the source exceeds the file-size limit, cannot use
    /// compact offsets, or cannot be represented as a compact syntax tree.
    pub fn parse(bytes: impl Into<std::sync::Arc<[u8]>>) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, ParseLimits::default(), &DialectRegistry::EMPTY)
    }
    /// Parses bytes with caller-supplied resource limits and generic syntax.
    ///
    /// The same limits are retained for [`Self::apply_text_edits`] and semantic
    /// lowering. Recoverable lexer and syntax problems appear as diagnostics.
    ///
    /// # Errors
    ///
    /// Returns the fatal errors described by [`Self::parse`].
    pub fn parse_with_limits(
        bytes: impl Into<std::sync::Arc<[u8]>>,
        limits: ParseLimits,
    ) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, limits, &DialectRegistry::EMPTY)
    }
    /// Parses bytes with default limits and registered custom operation syntax.
    ///
    /// Pass the same registry to semantic lowering, verification, and custom
    /// printing so every stage uses the same dialect contract.
    ///
    /// # Errors
    ///
    /// Returns the fatal errors described by [`Self::parse`].
    pub fn parse_with_registry(
        bytes: impl Into<std::sync::Arc<[u8]>>,
        registry: &DialectRegistry,
    ) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, ParseLimits::default(), registry)
    }
    /// Parses bytes with explicit limits and a dialect registry.
    ///
    /// A successful result is lossless even when it contains diagnostics.
    /// Registered parse callbacks may add syntax errors but do not replace the
    /// generic recovery model.
    ///
    /// # Errors
    ///
    /// Returns [`ParseFileError::ResourceLimit`] when `max_file_bytes` is
    /// exceeded, [`ParseFileError::Source`] when compact source offsets cannot
    /// represent the input, or [`ParseFileError::Syntax`] when CST compaction
    /// rejects the event stream.
    pub fn parse_with_limits_and_registry(
        bytes: impl Into<std::sync::Arc<[u8]>>,
        limits: ParseLimits,
        registry: &DialectRegistry,
    ) -> Result<Self, ParseFileError> {
        let bytes = bytes.into();
        if bytes.len() > limits.max_file_bytes {
            return Err(ParseFileError::ResourceLimit(ResourceLimitError {
                limit: limits.max_file_bytes,
                actual: bytes.len(),
            }));
        }
        let source = Source::new(bytes).map_err(ParseFileError::Source)?;
        let lexed = lex_with_limits(
            &source,
            LexerLimits {
                max_file_bytes: limits.max_file_bytes,
                max_tokens: limits.max_tokens,
            },
        );
        let lexer_diagnostics = lexed.diagnostics().to_vec();
        let syntax = parse_operations_with_registry(
            &lexed,
            source.bytes(),
            registry,
            ParserLimits {
                max_delimiter_depth: limits.max_delimiter_depth,
                max_payload_bytes: limits.max_payload_bytes,
                max_numeric_literal_bytes: limits.max_numeric_literal_bytes,
            },
        )
        .map_err(ParseFileError::Syntax)?;
        Ok(Self {
            source,
            lexer_diagnostics,
            syntax,
            limits,
        })
    }
    pub fn source(&self) -> &Source {
        &self.source
    }
    pub fn lexer_diagnostics(&self) -> &[LexDiagnostic] {
        &self.lexer_diagnostics
    }
    pub fn syntax(&self) -> &ParsedSyntax {
        &self.syntax
    }
    pub fn max_attribute_depth(&self) -> usize {
        self.limits.max_attribute_depth
    }
    pub fn max_alias_expansion_depth(&self) -> usize {
        self.limits.max_alias_expansion_depth
    }
    pub fn original_bytes(&self) -> &[u8] {
        self.source.bytes()
    }
    pub fn write_original<W: std::io::Write>(&self, sink: &mut W) -> std::io::Result<()> {
        sink.write_all(self.original_bytes())
    }
    /// Applies non-overlapping byte-range edits and reparses the complete result.
    ///
    /// Ranges refer to this file's original bytes. The returned file preserves
    /// this file's complete [`ParseLimits`].
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-bounds or overlapping range, an
    /// unrepresentable output length, or a fatal error while reparsing.
    pub fn apply_text_edits(&self, edits: &[TextEdit]) -> Result<Self, ApplyTextEditsError> {
        self.apply_text_edits_with_registry(edits, &DialectRegistry::EMPTY)
    }
    /// Applies text edits and reparses with explicit custom syntax.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::apply_text_edits`].
    pub fn apply_text_edits_with_registry(
        &self,
        edits: &[TextEdit],
        registry: &DialectRegistry,
    ) -> Result<Self, ApplyTextEditsError> {
        let mut ordered = edits.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|edit| (edit.range.start(), edit.range.end()));
        let mut end = 0usize;
        for edit in &ordered {
            let start = edit.range.start() as usize;
            let next_end = edit.range.end() as usize;
            if next_end > self.original_bytes().len() {
                return Err(ApplyTextEditsError::Edit(TextEditError::OutOfBounds(
                    edit.range,
                )));
            }
            if start < end {
                return Err(ApplyTextEditsError::Edit(TextEditError::Overlapping(
                    edit.range,
                )));
            }
            end = next_end;
        }
        let output_size = checked_text_edit_output_size(
            self.original_bytes().len(),
            ordered
                .iter()
                .map(|edit| (edit.range.len() as usize, edit.replacement.len())),
        )
        .ok_or(ApplyTextEditsError::OutputSizeOverflow)?;
        let mut bytes = Vec::with_capacity(output_size);
        let mut cursor = 0usize;
        for edit in ordered {
            let start = edit.range.start() as usize;
            let end = edit.range.end() as usize;
            bytes.extend_from_slice(&self.original_bytes()[cursor..start]);
            bytes.extend_from_slice(&edit.replacement);
            cursor = end;
        }
        bytes.extend_from_slice(&self.original_bytes()[cursor..]);
        Self::parse_with_limits_and_registry(bytes, self.limits, registry)
            .map_err(ApplyTextEditsError::Parse)
    }
}

fn checked_text_edit_output_size(
    original_size: usize,
    contributions: impl IntoIterator<Item = (usize, usize)>,
) -> Option<usize> {
    contributions
        .into_iter()
        .try_fold(original_size, |size, (removed_bytes, replacement_bytes)| {
            size.checked_sub(removed_bytes)?
                .checked_add(replacement_bytes)
        })
}

#[cfg(test)]
mod text_edit_size_tests {
    use super::{ParseLimits, ParsedFile, TextEdit, checked_text_edit_output_size};
    use crate::source::TextRange;
    use std::sync::Arc;

    #[test]
    fn overflowing_replacement_total_is_rejected() {
        assert_eq!(
            checked_text_edit_output_size(0, [(0, usize::MAX), (0, 1)]),
            None
        );
    }

    #[test]
    fn reparsed_file_retains_every_parse_limit() {
        let limits = ParseLimits {
            max_file_bytes: 128,
            max_tokens: 32,
            max_delimiter_depth: 7,
            max_payload_bytes: 11,
            max_numeric_literal_bytes: 13,
            max_attribute_depth: 17,
            max_alias_expansion_depth: 19,
        };
        let original =
            ParsedFile::parse_with_limits(b"\"old\"() : () -> ()".as_slice(), limits).unwrap();
        let edited = original
            .apply_text_edits(&[TextEdit {
                range: TextRange::new(1, 4).unwrap(),
                replacement: Arc::from(b"new".as_slice()),
            }])
            .unwrap();

        assert_eq!(edited.limits, limits);
        assert_eq!(original.original_bytes(), b"\"old\"() : () -> ()");
        assert_eq!(edited.original_bytes(), b"\"new\"() : () -> ()");
    }
}

/// Failure while validating or reparsing a batch of source edits.
#[derive(Debug)]
pub enum ApplyTextEditsError {
    Edit(TextEditError),
    OutputSizeOverflow,
    Parse(ParseFileError),
}

impl std::fmt::Display for ApplyTextEditsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Edit(error) => write!(f, "invalid text edit: {error}"),
            Self::OutputSizeOverflow => f.write_str("text edit output size cannot be represented"),
            Self::Parse(error) => write!(f, "edited source could not be parsed: {error}"),
        }
    }
}

impl std::error::Error for ApplyTextEditsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Edit(error) => Some(error),
            Self::Parse(error) => Some(error),
            Self::OutputSizeOverflow => None,
        }
    }
}

/// Fatal failure while constructing an owning parse result.
///
/// Recoverable lexer and grammar problems are diagnostics on [`ParsedFile`],
/// not variants of this error.
#[derive(Debug)]
pub enum ParseFileError {
    Source(SourceError),
    Syntax(CompactError),
    ResourceLimit(ResourceLimitError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimitError {
    pub limit: usize,
    pub actual: usize,
}

impl std::fmt::Display for ParseFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => write!(f, "invalid source: {error}"),
            Self::Syntax(error) => write!(f, "syntax tree construction failed: {error}"),
            Self::ResourceLimit(error) => write!(
                f,
                "file size {} exceeds limit {}",
                error.actual, error.limit
            ),
        }
    }
}

impl std::error::Error for ParseFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Syntax(error) => Some(error),
            Self::ResourceLimit(error) => Some(error),
        }
    }
}

impl std::fmt::Display for ResourceLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "file size {} exceeds limit {}", self.actual, self.limit)
    }
}

impl std::error::Error for ResourceLimitError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseDiagnostic {
    range: TextRange,
    kind: ParseDiagnosticKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseDiagnosticKind {
    Syntax,
    UnknownCustomOperation,
    ProgressLimit,
    DepthLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserLimits {
    pub max_delimiter_depth: usize,
    pub max_payload_bytes: usize,
    pub max_numeric_literal_bytes: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_delimiter_depth: 64,
            max_payload_bytes: 16 * 1024 * 1024,
            max_numeric_literal_bytes: 4096,
        }
    }
}

impl ParseDiagnostic {
    pub fn range(self) -> TextRange {
        self.range
    }
    pub fn kind(self) -> ParseDiagnosticKind {
        self.kind
    }
}

mod custom;
mod grammar;
mod syntax;

pub use custom::DialectParser;
use custom::shaped_operation;
#[doc(hidden)]
pub use grammar::parse_brace_fixture;
use grammar::{Parser, close_for};
pub use grammar::{
    parse_generic_operations, parse_generic_operations_with_limits, parse_operations_with_registry,
};
use syntax::is_trivia;
pub use syntax::{
    AffineSyntax, AttributeDictSyntax, BaseSyntax, BlockArgumentSyntax, BlockSyntax, FileSyntax,
    FunctionTypeSyntax, OpaqueBodySyntax, OperandUseSyntax, OperationComponentSyntax,
    OperationSyntax, ParsedSyntax, PayloadAttributeSyntax, PropertyDictSyntax, RegionSyntax,
    ResultGroupSyntax, SuccessorArgumentsSyntax, SuccessorSyntax, SyntaxNode,
    TrailingLocationSyntax, TypeSyntax, WideNumberSyntax,
};

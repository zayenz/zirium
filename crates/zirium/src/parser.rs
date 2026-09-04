use crate::{
    dialect::{DialectRegistry, OperationDescriptor, OperationShape},
    lexer::{Diagnostic as LexDiagnostic, Lexed, LexerLimits, TokenKind, lex_with_limits},
    representation::{
        CompactError, CompletedMarker, EventBuilder, Marker, NodeId, SyntaxKind, SyntaxTree,
    },
    source::{Source, SourceError, TextRange},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEdit {
    pub range: TextRange,
    pub replacement: std::sync::Arc<[u8]>,
}

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
#[derive(Debug)]
pub struct ParsedFile {
    source: Source,
    lexer_diagnostics: Vec<LexDiagnostic>,
    syntax: ParsedSyntax,
    max_attribute_depth: usize,
    max_alias_expansion_depth: usize,
}

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
    /// Parses and owns the input bytes using default limits and no custom dialect syntax.
    pub fn parse(bytes: impl Into<std::sync::Arc<[u8]>>) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, ParseLimits::default(), &DialectRegistry::EMPTY)
    }
    pub fn parse_with_limits(
        bytes: impl Into<std::sync::Arc<[u8]>>,
        limits: ParseLimits,
    ) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, limits, &DialectRegistry::EMPTY)
    }
    pub fn parse_with_registry(
        bytes: impl Into<std::sync::Arc<[u8]>>,
        registry: &DialectRegistry,
    ) -> Result<Self, ParseFileError> {
        Self::parse_with_limits_and_registry(bytes, ParseLimits::default(), registry)
    }
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
            max_attribute_depth: limits.max_attribute_depth,
            max_alias_expansion_depth: limits.max_alias_expansion_depth,
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
        self.max_attribute_depth
    }
    pub fn max_alias_expansion_depth(&self) -> usize {
        self.max_alias_expansion_depth
    }
    pub fn original_bytes(&self) -> &[u8] {
        self.source.bytes()
    }
    pub fn write_original<W: std::io::Write>(&self, sink: &mut W) -> std::io::Result<()> {
        sink.write_all(self.original_bytes())
    }
    /// Applies non-overlapping byte-range edits and reparses the complete result.
    pub fn apply_text_edits(&self, edits: &[TextEdit]) -> Result<Self, ApplyTextEditsError> {
        self.apply_text_edits_with_registry(edits, &DialectRegistry::EMPTY)
    }
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
        Self::parse_with_registry(bytes, registry).map_err(ApplyTextEditsError::Parse)
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
    use super::checked_text_edit_output_size;

    #[test]
    fn overflowing_replacement_total_is_rejected() {
        assert_eq!(
            checked_text_edit_output_size(0, [(0, usize::MAX), (0, 1)]),
            None
        );
    }
}

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

#[derive(Debug)]
pub struct ParsedSyntax {
    tree: std::sync::Arc<SyntaxTree>,
    diagnostics: Vec<ParseDiagnostic>,
}

impl ParsedSyntax {
    pub fn tree(&self) -> &SyntaxTree {
        &self.tree
    }
    pub(crate) fn shared_tree(&self) -> std::sync::Arc<SyntaxTree> {
        self.tree.clone()
    }
    pub fn diagnostics(&self) -> &[ParseDiagnostic] {
        &self.diagnostics
    }
    pub fn file(&self) -> FileSyntax<'_> {
        FileSyntax { tree: &self.tree }
    }
}

#[derive(Clone, Copy)]
pub struct FileSyntax<'a> {
    tree: &'a SyntaxTree,
}

impl<'a> FileSyntax<'a> {
    pub fn syntax(self) -> impl Iterator<Item = SyntaxNode<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .map(|id| SyntaxNode {
                tree: self.tree,
                id,
            })
    }
    pub fn typed_syntax(self) -> impl Iterator<Item = BaseSyntax<'a>> {
        self.syntax().map(SyntaxNode::typed)
    }
    pub fn node(self, id: NodeId) -> Option<SyntaxNode<'a>> {
        self.tree.kind(id)?;
        Some(SyntaxNode {
            tree: self.tree,
            id,
        })
    }
    pub fn operations(self) -> impl Iterator<Item = OperationSyntax<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .filter(|id| {
                matches!(
                    self.tree.kind(*id),
                    Some(
                        SyntaxKind::Operation
                            | SyntaxKind::DialectOperation
                            | SyntaxKind::UnparsedCustomOperation
                    )
                )
            })
            .map(|id| OperationSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn operation(self, id: NodeId) -> Option<OperationSyntax<'a>> {
        matches!(
            self.tree.kind(id),
            Some(
                SyntaxKind::Operation
                    | SyntaxKind::DialectOperation
                    | SyntaxKind::UnparsedCustomOperation
            )
        )
        .then_some(OperationSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn alias_definitions(self) -> impl Iterator<Item = SyntaxNode<'a>> {
        typed(self.tree, SyntaxKind::AliasDefinition).map(|id| SyntaxNode {
            tree: self.tree,
            id,
        })
    }
    pub fn regions(self) -> impl Iterator<Item = RegionSyntax<'a>> {
        typed(self.tree, SyntaxKind::Region).map(|id| RegionSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn nodes(self, kind: SyntaxKind) -> impl Iterator<Item = NodeId> + 'a {
        typed(self.tree, kind)
    }
    pub fn shaped_types(self) -> impl Iterator<Item = TypeSyntax<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .filter(|id| {
                matches!(
                    self.tree.kind(*id),
                    Some(
                        SyntaxKind::TupleType
                            | SyntaxKind::TensorType
                            | SyntaxKind::VectorType
                            | SyntaxKind::MemRefType
                    )
                )
            })
            .map(|id| TypeSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn affine_values(self) -> impl Iterator<Item = AffineSyntax<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .filter(|id| {
                matches!(
                    self.tree.kind(*id),
                    Some(SyntaxKind::AffineMap | SyntaxKind::IntegerSet)
                )
            })
            .map(|id| AffineSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn function_types(self) -> impl Iterator<Item = FunctionTypeSyntax<'a>> {
        typed(self.tree, SyntaxKind::FunctionType).map(|id| FunctionTypeSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn payload_attributes(self) -> impl Iterator<Item = PayloadAttributeSyntax<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .filter(|id| {
                matches!(
                    self.tree.kind(*id),
                    Some(
                        SyntaxKind::DenseElementsAttribute
                            | SyntaxKind::SparseElementsAttribute
                            | SyntaxKind::DenseResourceElementsAttribute
                    )
                )
            })
            .map(|id| PayloadAttributeSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn opaque_bodies(self) -> impl Iterator<Item = OpaqueBodySyntax<'a>> {
        self.tree
            .subtree(self.tree.root())
            .into_iter()
            .flatten()
            .filter(|id| {
                matches!(
                    self.tree.kind(*id),
                    Some(SyntaxKind::OpaqueAttributeBody | SyntaxKind::OpaqueTypeBody)
                )
            })
            .map(|id| OpaqueBodySyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn wide_numbers(self) -> impl Iterator<Item = WideNumberSyntax<'a>> {
        typed(self.tree, SyntaxKind::WideNumber).map(|id| WideNumberSyntax {
            tree: self.tree,
            id,
        })
    }
}

/// A borrowed view of any owned base syntax node.
///
/// The view is deliberately structural: it exposes the node's tag, children,
/// range, and error state without assigning semantic meaning to its contents.
#[derive(Clone, Copy)]
pub struct SyntaxNode<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> SyntaxNode<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    pub fn kind(self) -> SyntaxKind {
        self.tree
            .kind(self.id)
            .expect("borrowed syntax nodes always refer to an existing node")
    }
    pub fn text_range(self) -> Option<TextRange> {
        self.tree.text_range(self.id)
    }
    pub fn has_error(self) -> bool {
        self.tree.has_error(self.id).unwrap_or(true)
    }
    pub fn children(self) -> impl Iterator<Item = SyntaxNode<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .map(|id| SyntaxNode {
                tree: self.tree,
                id,
            })
    }
    pub fn typed(self) -> BaseSyntax<'a> {
        BaseSyntax::from(self)
    }
}

macro_rules! base_syntax_views {
    ($($variant:ident),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub enum BaseSyntax<'a> {
            $($variant(SyntaxNode<'a>)),+
        }

        impl<'a> From<SyntaxNode<'a>> for BaseSyntax<'a> {
            fn from(node: SyntaxNode<'a>) -> Self {
                match node.kind() {
                    $(SyntaxKind::$variant => Self::$variant(node)),+
                }
            }
        }

        impl<'a> BaseSyntax<'a> {
            pub fn node(self) -> SyntaxNode<'a> {
                match self {
                    $(Self::$variant(node) => node),+
                }
            }
        }
    };
}

base_syntax_views!(
    File,
    FileMetadata,
    AliasDefinition,
    Operation,
    DialectOperation,
    ArithConstantValue,
    UnparsedCustomOperation,
    Region,
    Block,
    Result,
    ResultGroup,
    ResultNumber,
    Operand,
    OperandUse,
    SuccessorList,
    Successor,
    SuccessorArguments,
    BlockLabel,
    BlockArgumentList,
    BlockArgument,
    PropertyDict,
    AttributeDict,
    Attribute,
    IntegerType,
    FloatType,
    IndexType,
    IntegerAttribute,
    FloatAttribute,
    BooleanAttribute,
    StringAttribute,
    TypeAttribute,
    AttributeAlias,
    TypeAlias,
    SymbolReference,
    ArrayAttribute,
    DictionaryAttribute,
    LocationAttribute,
    UnknownLocation,
    FileLineColLocation,
    NameLocation,
    CallSiteLocation,
    FusedLocation,
    TrailingLocation,
    FunctionType,
    TupleType,
    TensorType,
    VectorType,
    MemRefType,
    ShapedDimension,
    TensorEncoding,
    MemRefLayout,
    MemRefMemorySpace,
    StridedLayout,
    AffineMap,
    IntegerSet,
    AffineExpression,
    AffineConstraint,
    DenseElementsAttribute,
    SparseElementsAttribute,
    DenseResourceElementsAttribute,
    WideNumber,
    OpaqueAttribute,
    OpaqueAttributeBody,
    OpaqueType,
    OpaqueTypeBody,
    Error,
);

#[derive(Clone, Copy)]
pub struct PayloadAttributeSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> PayloadAttributeSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    pub fn kind(self) -> Option<SyntaxKind> {
        self.tree.kind(self.id)
    }
    pub fn payload_range(self) -> Option<TextRange> {
        delimited_inner_range(self.tree, self.id)
    }
    pub fn handle_range(self) -> Option<TextRange> {
        (self.tree.kind(self.id) == Some(SyntaxKind::DenseResourceElementsAttribute))
            .then(|| delimited_inner_range(self.tree, self.id))?
    }
    pub fn type_range(self) -> Option<TextRange> {
        suffix_type_range(self.tree, self.id)
    }
}

#[derive(Clone, Copy)]
pub struct OpaqueBodySyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> OpaqueBodySyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    pub fn kind(self) -> Option<SyntaxKind> {
        self.tree.kind(self.id)
    }
    pub fn body_range(self) -> Option<TextRange> {
        delimited_inner_range(self.tree, self.id)
    }
}

#[derive(Clone, Copy)]
pub struct WideNumberSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> WideNumberSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    pub fn literal_range(self) -> Option<TextRange> {
        self.tree
            .tokens(self.id)?
            .iter()
            .find(|token| token.kind() == TokenKind::WideInteger)
            .map(|token| token.range())
    }
    pub fn type_range(self) -> Option<TextRange> {
        suffix_type_range(self.tree, self.id)
    }
}

fn delimited_inner_range(tree: &SyntaxTree, id: NodeId) -> Option<TextRange> {
    let tokens = tree.tokens(id)?;
    let open = tokens.iter().position(|t| t.kind() == TokenKind::Less)?;
    let mut stack = vec![TokenKind::Greater];
    let mut close = None;
    for (index, token) in tokens.iter().enumerate().skip(open + 1) {
        let kind = token.kind();
        if stack.last() == Some(&kind) {
            stack.pop();
            if stack.is_empty() {
                close = Some(index);
                break;
            }
        } else if let Some(expected) = close_for(kind) {
            stack.push(expected);
        }
    }
    let close = close?;
    TextRange::new(tokens[open].range().end(), tokens[close].range().start())
}

fn suffix_type_range(tree: &SyntaxTree, id: NodeId) -> Option<TextRange> {
    let tokens = tree.tokens(id)?;
    let colon = tokens.iter().rposition(|t| t.kind() == TokenKind::Colon)?;
    let first = tokens[colon + 1..].iter().find(|t| !is_trivia(t.kind()))?;
    let last = tokens.iter().rev().find(|t| !is_trivia(t.kind()))?;
    TextRange::new(first.range().start(), last.range().end())
}

fn is_trivia(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Whitespace | TokenKind::LineComment)
}

#[derive(Clone, Copy)]
pub struct FunctionTypeSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> FunctionTypeSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
}

#[derive(Clone, Copy)]
pub struct TypeSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> TypeSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
}

#[derive(Clone, Copy)]
pub struct AffineSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> AffineSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
}

#[derive(Clone, Copy)]
pub struct OperationSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> OperationSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    /// Returns the token range of the operation mnemonic.
    pub fn mnemonic_range(self) -> Option<TextRange> {
        let tokens = self.tree.tokens(self.id)?;
        let mut index = tokens.iter().position(|token| !is_trivia(token.kind()))?;
        if tokens[index].kind() == TokenKind::PercentIdentifier {
            index = tokens
                .iter()
                .position(|token| token.kind() == TokenKind::Equal)?
                + 1;
            index += tokens[index..]
                .iter()
                .position(|token| !is_trivia(token.kind()))?;
        }
        matches!(
            tokens[index].kind(),
            TokenKind::BareIdentifier | TokenKind::String
        )
        .then(|| tokens[index].range())
    }
    /// Returns a leading symbol token from the operation header, if present.
    pub fn leading_symbol_range(self) -> Option<TextRange> {
        let mnemonic = self.mnemonic_range()?;
        let operation_end = self.tree.text_range(self.id)?.end();
        let header_end = self
            .tree
            .children(self.id)
            .into_iter()
            .flatten()
            .filter(|child| {
                matches!(
                    self.tree.kind(*child),
                    Some(SyntaxKind::AttributeDict | SyntaxKind::Region)
                )
            })
            .filter_map(|child| self.tree.text_range(child).map(|range| range.start()))
            .min()
            .unwrap_or(operation_end);
        self.tree.tokens(self.id)?.iter().find_map(|token| {
            (token.range().start() >= mnemonic.end()
                && token.range().end() <= header_end
                && token.kind() == TokenKind::AtIdentifier)
                .then(|| token.range())
        })
    }
    /// Returns the optional visibility keyword immediately before a leading symbol.
    pub fn visibility_range(self) -> Option<TextRange> {
        let mnemonic = self.mnemonic_range()?;
        let symbol = self.leading_symbol_range()?;
        self.tree.tokens(self.id)?.iter().find_map(|token| {
            (token.range().start() >= mnemonic.end()
                && token.range().end() <= symbol.start()
                && token.kind() == TokenKind::BareIdentifier)
                .then(|| token.range())
        })
    }
    pub fn argument_list_range(self) -> Option<TextRange> {
        child_of_kind(self.tree, self.id, SyntaxKind::BlockArgumentList)
            .and_then(|id| self.tree.text_range(id))
    }
    pub fn function_type_range(self) -> Option<TextRange> {
        child_of_kind(self.tree, self.id, SyntaxKind::FunctionType)
            .and_then(|id| self.tree.text_range(id))
    }
    /// Returns the result portion of a func-like signature, including its arrow.
    pub fn function_result_range(self, source: &[u8]) -> Option<TextRange> {
        let arguments = self.argument_list_range()?;
        let tokens = self.tree.tokens(self.id)?;
        let boundary = self
            .tree
            .children(self.id)?
            .filter_map(|child| {
                let range = self.tree.text_range(child)?;
                match self.tree.kind(child) {
                    Some(SyntaxKind::Region) => Some(range.start()),
                    Some(SyntaxKind::AttributeDict)
                        if tokens
                            .iter()
                            .rev()
                            .find(|token| {
                                token.range().end() <= range.start() && !is_trivia(token.kind())
                            })
                            .and_then(|token| {
                                source.get(
                                    token.range().start() as usize..token.range().end() as usize,
                                )
                            })
                            == Some(b"attributes") =>
                    {
                        Some(range.start())
                    }
                    _ => None,
                }
            })
            .min()
            .unwrap_or(self.tree.text_range(self.id)?.end());
        let arrow = tokens.iter().find(|token| {
            token.range().start() >= arguments.end()
                && token.range().end() <= boundary
                && token.kind() == TokenKind::Arrow
        })?;
        let mut last = tokens.iter().rev().find(|token| {
            token.range().start() >= arrow.range().start()
                && token.range().end() <= boundary
                && !is_trivia(token.kind())
        })?;
        if source.get(last.range().start() as usize..last.range().end() as usize)
            == Some(b"attributes")
        {
            last = tokens.iter().rev().find(|token| {
                token.range().start() >= arrow.range().start()
                    && token.range().end() <= last.range().start()
                    && !is_trivia(token.kind())
            })?;
        }
        TextRange::new(arrow.range().start(), last.range().end())
    }
    pub fn components(self) -> impl Iterator<Item = OperationComponentSyntax<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .map(|id| OperationComponentSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn results(self) -> impl Iterator<Item = ResultGroupSyntax<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .filter(|id| self.tree.kind(*id) == Some(SyntaxKind::Result))
            .flat_map(move |list| children_of_kind(self.tree, list, SyntaxKind::ResultGroup))
            .map(|id| ResultGroupSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn operands(self) -> impl Iterator<Item = OperandUseSyntax<'a>> {
        children_of_kind(self.tree, self.id, SyntaxKind::Operand)
            .flat_map(move |operand| children_of_kind(self.tree, operand, SyntaxKind::OperandUse))
            .map(|id| OperandUseSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn arguments(self) -> impl Iterator<Item = BlockArgumentSyntax<'a>> {
        children_of_kind(self.tree, self.id, SyntaxKind::BlockArgumentList)
            .flat_map(move |list| children_of_kind(self.tree, list, SyntaxKind::BlockArgument))
            .map(|id| BlockArgumentSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn successors(self) -> impl Iterator<Item = SuccessorSyntax<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .filter(|id| self.tree.kind(*id) == Some(SyntaxKind::SuccessorList))
            .flat_map(move |list| children_of_kind(self.tree, list, SyntaxKind::Successor))
            .map(|id| SuccessorSyntax {
                tree: self.tree,
                id,
            })
    }
    pub fn regions(self) -> impl Iterator<Item = RegionSyntax<'a>> {
        children_of_kind(self.tree, self.id, SyntaxKind::Region).map(|id| RegionSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn properties(self) -> Option<PropertyDictSyntax<'a>> {
        child_of_kind(self.tree, self.id, SyntaxKind::PropertyDict).map(|id| PropertyDictSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn attributes(self) -> Option<AttributeDictSyntax<'a>> {
        child_of_kind(self.tree, self.id, SyntaxKind::AttributeDict).map(|id| AttributeDictSyntax {
            tree: self.tree,
            id,
        })
    }
    pub fn trailing_location(self) -> Option<TrailingLocationSyntax<'a>> {
        child_of_kind(self.tree, self.id, SyntaxKind::TrailingLocation).map(|id| {
            TrailingLocationSyntax {
                tree: self.tree,
                id,
            }
        })
    }
}

macro_rules! borrowed_view {
    ($name:ident) => {
        #[derive(Clone, Copy)]
        pub struct $name<'a> {
            tree: &'a SyntaxTree,
            id: NodeId,
        }
        impl<'a> $name<'a> {
            pub fn id(self) -> NodeId {
                self.id
            }
            pub fn tree(self) -> &'a SyntaxTree {
                self.tree
            }
        }
    };
}
borrowed_view!(OperationComponentSyntax);
borrowed_view!(ResultGroupSyntax);
borrowed_view!(OperandUseSyntax);
borrowed_view!(BlockArgumentSyntax);
borrowed_view!(SuccessorArgumentsSyntax);
borrowed_view!(PropertyDictSyntax);
borrowed_view!(AttributeDictSyntax);
borrowed_view!(TrailingLocationSyntax);

impl BlockArgumentSyntax<'_> {
    pub fn attribute_range(self) -> Option<TextRange> {
        child_of_kind(self.tree, self.id, SyntaxKind::AttributeDict)
            .and_then(|id| self.tree.text_range(id))
    }
}

impl<'a> SuccessorArgumentsSyntax<'a> {
    pub fn arguments(self) -> impl Iterator<Item = BlockArgumentSyntax<'a>> {
        children_of_kind(self.tree, self.id, SyntaxKind::BlockArgument).map(|id| {
            BlockArgumentSyntax {
                tree: self.tree,
                id,
            }
        })
    }
}

impl ResultGroupSyntax<'_> {
    pub fn number(self) -> Option<NodeId> {
        child_of_kind(self.tree, self.id, SyntaxKind::ResultNumber)
    }
}

#[derive(Clone, Copy)]
pub struct SuccessorSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}
impl<'a> SuccessorSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn tree(self) -> &'a SyntaxTree {
        self.tree
    }
    pub fn arguments_group(self) -> Option<SuccessorArgumentsSyntax<'a>> {
        child_of_kind(self.tree, self.id, SyntaxKind::SuccessorArguments).map(|id| {
            SuccessorArgumentsSyntax {
                tree: self.tree,
                id,
            }
        })
    }
    pub fn arguments(self) -> impl Iterator<Item = BlockArgumentSyntax<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .filter(|id| self.tree.kind(*id) == Some(SyntaxKind::SuccessorArguments))
            .flat_map(move |list| children_of_kind(self.tree, list, SyntaxKind::BlockArgument))
            .map(|id| BlockArgumentSyntax {
                tree: self.tree,
                id,
            })
    }
}

#[derive(Clone, Copy)]
pub struct RegionSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}

impl<'a> RegionSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn implicit_block(self) -> Option<NodeId> {
        self.tree
            .children(self.id)?
            .find(|id| self.tree.kind(*id) == Some(SyntaxKind::Block))
    }
    pub fn blocks(self) -> impl Iterator<Item = BlockSyntax<'a>> {
        children_of_kind(self.tree, self.id, SyntaxKind::Block).map(|id| BlockSyntax {
            tree: self.tree,
            id,
        })
    }
}

#[derive(Clone, Copy)]
pub struct BlockSyntax<'a> {
    tree: &'a SyntaxTree,
    id: NodeId,
}
impl<'a> BlockSyntax<'a> {
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn label(self) -> Option<NodeId> {
        child_of_kind(self.tree, self.id, SyntaxKind::BlockLabel)
    }
    pub fn arguments(self) -> impl Iterator<Item = BlockArgumentSyntax<'a>> {
        self.tree
            .children(self.id)
            .into_iter()
            .flatten()
            .filter(|id| self.tree.kind(*id) == Some(SyntaxKind::BlockLabel))
            .flat_map(move |label| {
                children_of_kind(self.tree, label, SyntaxKind::BlockArgumentList)
            })
            .flat_map(move |list| children_of_kind(self.tree, list, SyntaxKind::BlockArgument))
            .map(|id| BlockArgumentSyntax {
                tree: self.tree,
                id,
            })
    }
}

fn child_of_kind(tree: &SyntaxTree, parent: NodeId, kind: SyntaxKind) -> Option<NodeId> {
    tree.children(parent)?
        .find(|id| tree.kind(*id) == Some(kind))
}
fn children_of_kind(
    tree: &SyntaxTree,
    parent: NodeId,
    kind: SyntaxKind,
) -> impl Iterator<Item = NodeId> + '_ {
    tree.children(parent)
        .into_iter()
        .flatten()
        .filter(move |id| tree.kind(*id) == Some(kind))
}

fn typed(tree: &SyntaxTree, kind: SyntaxKind) -> impl Iterator<Item = NodeId> + '_ {
    tree.subtree(tree.root())
        .into_iter()
        .flatten()
        .filter(move |id| tree.kind(*id) == Some(kind))
}

/// Parses quoted generic operations into a lossless, syntactic CST.
pub fn parse_generic_operations(lexed: &Lexed) -> Result<ParsedSyntax, CompactError> {
    parse_generic_operations_with_limits(lexed, ParserLimits::default())
}

pub fn parse_generic_operations_with_limits(
    lexed: &Lexed,
    limits: ParserLimits,
) -> Result<ParsedSyntax, CompactError> {
    parse_operations_with_registry(lexed, &[], &DialectRegistry::EMPTY, limits)
}

pub fn parse_operations_with_registry(
    lexed: &Lexed,
    source: &[u8],
    registry: &DialectRegistry,
    limits: ParserLimits,
) -> Result<ParsedSyntax, CompactError> {
    let (builder, diagnostics) = produce_operation_events(lexed, source, registry, limits)?;
    Ok(ParsedSyntax {
        tree: std::sync::Arc::new(builder.finish(lexed.tokens().to_vec())?),
        diagnostics,
    })
}

fn produce_operation_events(
    lexed: &Lexed,
    source: &[u8],
    registry: &DialectRegistry,
    limits: ParserLimits,
) -> Result<(EventBuilder, Vec<ParseDiagnostic>), CompactError> {
    let mut parser = Parser {
        tokens: lexed.tokens(),
        position: 0,
        builder: EventBuilder::new(),
        diagnostics: Vec::new(),
        limits,
        nesting_depth: 0,
        source,
        registry,
    };
    let root = parser.builder.start();
    while !parser.at(TokenKind::Eof) {
        let before = parser.position;
        parser.trivia()?;
        if parser.at(TokenKind::FileMetadataBegin) {
            parser.file_metadata()?;
        } else if matches!(
            parser.current(),
            TokenKind::HashIdentifier | TokenKind::ExclamationIdentifier
        ) && parser.nth_nontrivia(1) == Some(TokenKind::Equal)
        {
            parser.alias_definition()?;
        } else if matches!(
            parser.current(),
            TokenKind::String | TokenKind::PercentIdentifier | TokenKind::BareIdentifier
        ) {
            parser.operation()?;
        } else if !parser.at(TokenKind::Eof) {
            parser.error_token()?;
        }
        parser.ensure_progress(before)?;
    }
    parser.bump()?;
    parser.builder.complete(root, SyntaxKind::File)?;
    Ok((parser.builder, parser.diagnostics))
}

#[cfg(test)]
mod parser_construction_benchmark;

struct Parser<'a> {
    tokens: &'a [crate::lexer::Token],
    position: usize,
    builder: EventBuilder,
    diagnostics: Vec<ParseDiagnostic>,
    limits: ParserLimits,
    nesting_depth: usize,
    source: &'a [u8],
    registry: &'a DialectRegistry,
}

/// Constrained token/CST access passed to registered syntax callbacks.
///
/// It deliberately exposes no semantic document or arena.
pub struct DialectParser<'a, 'registry> {
    parser: &'a mut Parser<'registry>,
    marker: Marker,
    descriptor: &'registry OperationDescriptor,
}

impl DialectParser<'_, '_> {
    pub fn parse_assembly_program(&mut self) -> Result<(), CompactError> {
        use crate::dialect::AssemblyProgram;
        match self
            .descriptor
            .assembly
            .expect("validated assembly program")
        {
            AssemblyProgram::Module => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::AtIdentifier) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::BareIdentifier)
                    && self.parser.current_text() == "attributes"
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    good &= self.parser.at(TokenKind::LBrace);
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.region()?;
                } else {
                    self.parser.diagnostic();
                    good = false;
                }
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::Function => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::BareIdentifier)
                    && matches!(self.parser.current_text(), "public" | "private" | "nested")
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::AtIdentifier)?;
                self.parser.trivia()?;
                good &= self
                    .parser
                    .block_argument_list(SyntaxKind::BlockArgumentList)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::Arrow) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    if self.parser.at(TokenKind::LParen) {
                        self.parser.type_list(0)?;
                    } else {
                        good &= self.parser.type_syntax(0)?;
                    }
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::BareIdentifier)
                    && self.parser.current_text() == "attributes"
                {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.region()?;
                }
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::Call => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                good &= self.parser.expect(TokenKind::AtIdentifier)?;
                self.parser.trivia()?;
                self.parser.operand_list()?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::Colon)?;
                self.parser.trivia()?;
                self.parser.function_type()?;
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::ConditionalBranch => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                good &= self.parse_operand()?;
                self.parser.trivia()?;
                good &= self.parser.expect(TokenKind::Comma)?;
                self.parser.trivia()?;
                let list = self.parser.builder.start();
                for index in 0..2 {
                    good &= self.parse_successor()?;
                    self.parser.trivia()?;
                    if index == 0 {
                        good &= self.parser.expect(TokenKind::Comma)?;
                        self.parser.trivia()?;
                    }
                }
                self.parser
                    .builder
                    .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                }
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::TypedAttribute => self.parse_zero_operand_constant(),
            AssemblyProgram::BinaryOperands => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                for index in 0..2 {
                    if self.parser.at(TokenKind::PercentIdentifier) {
                        let operand = self.parser.builder.start();
                        let use_marker = self.parser.builder.start();
                        self.parser.bump()?;
                        self.parser
                            .builder
                            .complete(use_marker, SyntaxKind::OperandUse)?;
                        self.parser.builder.complete(operand, SyntaxKind::Operand)?;
                    } else {
                        good = false;
                        self.parser.diagnostic();
                    }
                    self.parser.trivia()?;
                    if index == 0 {
                        good &= self.parser.expect(TokenKind::Comma)?;
                        self.parser.trivia()?;
                    }
                }
                if self.parser.at(TokenKind::BareIdentifier) {
                    if self.parser.current_text() == "overflow" {
                        good &= self.parse_overflow_flags()?;
                        self.parser.trivia()?;
                    } else {
                        self.parser.diagnostic();
                        good = false;
                        self.parser.bump()?;
                    }
                } else if !self.parser.at(TokenKind::LBrace) && !self.parser.at(TokenKind::Colon) {
                    self.parser.diagnostic();
                    good = false;
                    self.parser.bump()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                good &= self.parser.expect(TokenKind::Colon)?;
                self.parser.trivia()?;
                good &= self.parser.type_syntax(0)?;
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::OptionalTypedOperands => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                let mut operand_count = 0;
                while self.parser.at(TokenKind::PercentIdentifier) {
                    let operand = self.parser.builder.start();
                    let use_marker = self.parser.builder.start();
                    self.parser.bump()?;
                    self.parser
                        .builder
                        .complete(use_marker, SyntaxKind::OperandUse)?;
                    self.parser.builder.complete(operand, SyntaxKind::Operand)?;
                    operand_count += 1;
                    self.parser.trivia()?;
                    if !self.parser.at(TokenKind::Comma) {
                        break;
                    }
                    self.parser.bump()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                    self.parser.trivia()?;
                }
                if self.parser.at(TokenKind::Colon) {
                    self.parser.bump()?;
                    self.parser.trivia()?;
                    let type_count = self.parse_return_type_list()?;
                    if type_count != operand_count {
                        self.parser.diagnostic();
                        good = false;
                    }
                } else if operand_count != 0 {
                    self.parser.diagnostic();
                    good = false;
                }
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
            AssemblyProgram::TypedSuccessor => {
                let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
                self.parser.trivia()?;
                let list = self.parser.builder.start();
                let successor = self.parser.builder.start();
                good &= self.parser.expect(TokenKind::CaretIdentifier)?;
                self.parser.trivia()?;
                if self.parser.at(TokenKind::LParen) {
                    good &= self
                        .parser
                        .block_argument_list(SyntaxKind::SuccessorArguments)?;
                    self.parser.trivia()?;
                }
                self.parser
                    .builder
                    .complete_with_error(successor, SyntaxKind::Successor, !good)?;
                self.parser
                    .builder
                    .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
                if self.parser.at(TokenKind::LBrace) {
                    self.parser.attribute_dict()?;
                }
                self.parser.builder.complete_with_error(
                    self.marker,
                    self.descriptor.syntax_kind,
                    !good,
                )?;
                Ok(())
            }
        }
    }

    fn parse_operand(&mut self) -> Result<bool, CompactError> {
        let operand = self.parser.builder.start();
        let use_marker = self.parser.builder.start();
        let good = self.parser.expect(TokenKind::PercentIdentifier)?;
        self.parser
            .builder
            .complete_with_error(use_marker, SyntaxKind::OperandUse, !good)?;
        self.parser
            .builder
            .complete_with_error(operand, SyntaxKind::Operand, !good)?;
        Ok(good)
    }

    fn parse_successor(&mut self) -> Result<bool, CompactError> {
        let successor = self.parser.builder.start();
        let mut good = self.parser.expect(TokenKind::CaretIdentifier)?;
        self.parser.trivia()?;
        if self.parser.at(TokenKind::LParen) {
            good &= self
                .parser
                .block_argument_list(SyntaxKind::SuccessorArguments)?;
        }
        self.parser
            .builder
            .complete_with_error(successor, SyntaxKind::Successor, !good)?;
        Ok(good)
    }

    pub fn parse_zero_operand_constant(&mut self) -> Result<(), CompactError> {
        let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
        self.parser.trivia()?;
        let value = self.parser.builder.start();
        let attr_good = self.parser.constant_value()?;
        good &= attr_good;
        self.parser
            .builder
            .complete_with_error(value, SyntaxKind::ArithConstantValue, !good)?;
        self.parser.trivia()?;
        if self.parser.at(TokenKind::LBrace) {
            self.parser.attribute_dict()?;
            self.parser.trivia()?;
        }
        let colon_good = self.parser.expect(TokenKind::Colon)?;
        good &= colon_good;
        self.parser.trivia()?;
        let type_good = self.parser.type_syntax(0)?;
        good &= type_good;
        self.parser
            .builder
            .complete_with_error(self.marker, self.descriptor.syntax_kind, !good)?;
        Ok(())
    }

    fn parse_overflow_flags(&mut self) -> Result<bool, CompactError> {
        let mut good = self.parser.expect(TokenKind::BareIdentifier)?;
        self.parser.trivia()?;
        good &= self.parser.expect(TokenKind::Less)?;
        self.parser.trivia()?;
        let mut seen_nsw = false;
        let mut seen_nuw = false;
        let mut seen_none = false;
        let mut count = 0;
        loop {
            if !self.parser.at(TokenKind::BareIdentifier) {
                good = false;
                self.parser.diagnostic();
                break;
            }
            let flag = self.parser.current_text();
            let duplicate = match flag {
                "nsw" => std::mem::replace(&mut seen_nsw, true),
                "nuw" => std::mem::replace(&mut seen_nuw, true),
                "none" => std::mem::replace(&mut seen_none, true),
                _ => true,
            };
            if duplicate || (flag == "none" && count != 0) || (flag != "none" && seen_none) {
                good = false;
                self.parser.diagnostic();
            }
            self.parser.bump()?;
            self.parser.trivia()?;
            count += 1;
            if count > 2 {
                good = false;
                self.parser.diagnostic();
            }
            if !self.parser.at(TokenKind::Comma) {
                break;
            }
            self.parser.bump()?;
            self.parser.trivia()?;
        }
        if count == 0 {
            good = false;
        }
        good &= self.parser.expect(TokenKind::Greater)?;
        Ok(good)
    }

    fn parse_return_type_list(&mut self) -> Result<usize, CompactError> {
        if self.parser.at(TokenKind::LParen) {
            self.parser.bump()?;
            self.parser.trivia()?;
            let mut count = 0;
            while self.parser.at_type_start() {
                let good = self.parser.type_syntax(0)?;
                count += usize::from(good);
                self.parser.trivia()?;
                if !self.parser.at(TokenKind::Comma) {
                    break;
                }
                self.parser.bump()?;
                self.parser.trivia()?;
            }
            self.parser.expect(TokenKind::RParen)?;
            Ok(count)
        } else {
            let good = self.parser.type_syntax(0)?;
            Ok(usize::from(good))
        }
    }
}

const MAX_TYPE_DEPTH: usize = 64;

impl Parser<'_> {
    fn file_metadata(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        while !self.at(TokenKind::FileMetadataEnd) && !self.at(TokenKind::Eof) {
            self.bump()?;
        }
        let unterminated = self.at(TokenKind::Eof);
        if unterminated {
            self.diagnostic();
        } else {
            self.bump()?;
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::FileMetadata, unterminated)?;
        Ok(())
    }

    fn alias_definition(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        let is_type = self.at(TokenKind::ExclamationIdentifier);
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Equal)?;
        self.trivia()?;
        if is_type {
            // MLIR permits the documentary `type` keyword in type aliases.
            if self.at(TokenKind::BareIdentifier) {
                self.bump()?;
                self.trivia()?;
            }
            if self.at(TokenKind::LParen) {
                self.function_type()?;
            } else {
                good &= self.type_syntax(0)?;
            }
        } else {
            good &= self.attribute_value()?;
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::AliasDefinition, !good)?;
        Ok(())
    }

    fn operation(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        if self.at(TokenKind::PercentIdentifier) {
            let result_list = self.builder.start();
            loop {
                let result = self.builder.start();
                self.bump()?;
                self.trivia()?;
                if self.at(TokenKind::Colon) {
                    let number = self.builder.start();
                    self.bump()?;
                    self.trivia()?;
                    let good = self.expect(TokenKind::Integer)?;
                    self.builder
                        .complete_with_error(number, SyntaxKind::ResultNumber, !good)?;
                    self.trivia()?;
                }
                self.builder.complete(result, SyntaxKind::ResultGroup)?;
                if !self.at(TokenKind::Comma) {
                    break;
                }
                self.bump()?;
                self.trivia()?;
                if !self.at(TokenKind::PercentIdentifier) {
                    self.diagnostic();
                    break;
                }
            }
            self.expect(TokenKind::Equal)?;
            self.builder.complete(result_list, SyntaxKind::Result)?;
            self.trivia()?;
        }
        if self.at(TokenKind::BareIdentifier) {
            let range = self.tokens[self.position].range();
            let name = std::str::from_utf8(
                self.source
                    .get(range.start() as usize..range.end() as usize)
                    .unwrap_or_default(),
            )
            .unwrap_or("");
            if let Some(descriptor) = self.registry.custom_operation(name) {
                if descriptor.assembly.is_some() {
                    return DialectParser {
                        parser: self,
                        marker,
                        descriptor,
                    }
                    .parse_assembly_program();
                }
                if let Some(parse) = descriptor.parse {
                    return parse(&mut DialectParser {
                        parser: self,
                        marker,
                        descriptor,
                    });
                }
            }
            if let Some(shape) = self.registry.operation_shape(name) {
                return self.shaped_operation(marker, shape);
            }
            return self.unparsed_custom_operation(Some(marker));
        }
        self.expect(TokenKind::String)?;
        self.trivia()?;
        self.operand_list()?;
        self.trivia()?;
        if self.at(TokenKind::LBracket) {
            self.successor_list()?;
            self.trivia()?;
        }
        if self.at(TokenKind::Less) {
            self.property_dict()?;
            self.trivia()?;
        }
        if self.at(TokenKind::LParen) && self.nth_nontrivia(1) == Some(TokenKind::LBrace) {
            self.region_list()?;
            self.trivia()?;
        }
        if self.at(TokenKind::LBrace) {
            self.attribute_dict()?;
            self.trivia()?;
        }
        self.expect(TokenKind::Colon)?;
        self.trivia()?;
        self.function_type()?;
        self.trivia()?;
        if self.at(TokenKind::Loc) {
            let location = self.builder.start();
            let good = self.location_attribute()?;
            self.builder
                .complete_with_error(location, SyntaxKind::TrailingLocation, !good)?;
        }
        self.builder.complete(marker, SyntaxKind::Operation)?;
        Ok(())
    }

    fn shaped_operation(
        &mut self,
        marker: Marker,
        shape: OperationShape,
    ) -> Result<(), CompactError> {
        let mut good = self.expect(TokenKind::BareIdentifier)?;
        self.trivia()?;
        good &= self.expect(TokenKind::AtIdentifier)?;
        self.trivia()?;
        match shape {
            OperationShape::FuncLike => {
                good &= self.block_argument_list(SyntaxKind::BlockArgumentList)?;
                self.trivia()?;
                if self.at(TokenKind::Arrow) {
                    self.bump()?;
                    self.trivia()?;
                    if self.at(TokenKind::LParen) {
                        self.type_list(0)?;
                    } else {
                        good &= self.type_syntax(0)?;
                    }
                    self.trivia()?;
                }
                if self.at(TokenKind::BareIdentifier) && self.current_text() == "attributes" {
                    self.bump()?;
                    self.trivia()?;
                    self.attribute_dict()?;
                    self.trivia()?;
                }
                if self.at(TokenKind::LBrace) {
                    self.region()?;
                }
            }
            OperationShape::CallLike => {
                self.operand_list()?;
                self.trivia()?;
                if self.at(TokenKind::LBrace) {
                    self.attribute_dict()?;
                    self.trivia()?;
                }
                good &= self.expect(TokenKind::Colon)?;
                self.trivia()?;
                self.function_type()?;
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::DialectOperation, !good)?;
        Ok(())
    }

    fn operand_list(&mut self) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at(TokenKind::PercentIdentifier) {
            let operand = self.builder.start();
            let operand_use = self.builder.start();
            self.bump()?;
            if self.at(TokenKind::HashIdentifier) {
                self.bump()?;
            }
            self.builder.complete(operand_use, SyntaxKind::OperandUse)?;
            self.builder.complete(operand, SyntaxKind::Operand)?;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(())
    }

    fn successor_list(&mut self) -> Result<(), CompactError> {
        let list = self.builder.start();
        let mut good = self.expect(TokenKind::LBracket)?;
        self.trivia()?;
        while !self.at(TokenKind::RBracket)
            && !self.at(TokenKind::RBrace)
            && !self.at(TokenKind::Eof)
        {
            let successor = self.builder.start();
            let mut item_good = self.expect(TokenKind::CaretIdentifier)?;
            self.trivia()?;
            if self.at(TokenKind::Colon) {
                self.bump()?;
                self.trivia()?;
                item_good &= self.block_argument_list(SyntaxKind::SuccessorArguments)?;
            }
            self.builder
                .complete_with_error(successor, SyntaxKind::Successor, !item_good)?;
            good &= item_good;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::RBracket)?;
        self.builder
            .complete_with_error(list, SyntaxKind::SuccessorList, !good)?;
        Ok(())
    }

    fn property_dict(&mut self) -> Result<(), CompactError> {
        let marker = self.builder.start();
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        good &= self.dictionary_entries()?;
        self.trivia()?;
        good &= self.expect(TokenKind::Greater)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::PropertyDict, !good)?;
        Ok(())
    }

    fn region_list(&mut self) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        loop {
            self.region()?;
            self.trivia()?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(())
    }

    fn region(&mut self) -> Result<(), CompactError> {
        let region = self.builder.start();
        if self.nesting_depth >= self.limits.max_delimiter_depth {
            self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
            self.recover_balanced_region()?;
            self.builder
                .complete_with_error(region, SyntaxKind::Region, true)?;
            return Ok(());
        }
        self.nesting_depth += 1;
        self.expect(TokenKind::LBrace)?;
        self.trivia()?;
        let mut block = (!self.at(TokenKind::CaretIdentifier)).then(|| self.builder.start());
        loop {
            let before = self.position;
            self.trivia()?;
            if self.at(TokenKind::RBrace) || self.at(TokenKind::Eof) {
                break;
            }
            if self.at(TokenKind::CaretIdentifier) {
                if let Some(open) = block.take() {
                    self.builder.complete(open, SyntaxKind::Block)?;
                }
                let labeled = self.builder.start();
                let label = self.builder.start();
                self.bump()?;
                self.trivia()?;
                if self.at(TokenKind::LParen) {
                    self.block_argument_list(SyntaxKind::BlockArgumentList)?;
                    self.trivia()?;
                }
                self.expect(TokenKind::Colon)?;
                self.builder.complete(label, SyntaxKind::BlockLabel)?;
                block = Some(labeled);
                continue;
            }
            if self.at(TokenKind::String)
                || self.at(TokenKind::PercentIdentifier)
                || self.at(TokenKind::BareIdentifier)
            {
                self.operation()?;
            } else {
                self.error_token()?;
            }
            self.ensure_progress(before)?;
        }
        if let Some(open) = block {
            self.builder.complete(open, SyntaxKind::Block)?;
        }
        self.expect(TokenKind::RBrace)?;
        self.nesting_depth -= 1;
        self.builder.complete(region, SyntaxKind::Region)?;
        Ok(())
    }

    fn block_argument_list(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let list = self.builder.start();
        let mut good = self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at(TokenKind::PercentIdentifier) {
            let argument = self.builder.start();
            self.bump()?;
            self.trivia()?;
            good &= self.expect(TokenKind::Colon)?;
            self.trivia()?;
            good &= self.type_syntax(0)?;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if self.at(TokenKind::Loc) {
                good &= self.location_attribute()?;
                self.trivia()?;
            }
            self.builder
                .complete_with_error(argument, SyntaxKind::BlockArgument, !good)?;
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::RParen)?;
        self.builder.complete_with_error(list, kind, !good)?;
        Ok(good)
    }

    fn attribute_dict(&mut self) -> Result<(), CompactError> {
        let dict = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBrace, TokenKind::RBrace)? {
            self.builder
                .complete_with_error(dict, SyntaxKind::AttributeDict, true)?;
            return Ok(());
        }
        let bad = !self.dictionary_entries()?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(dict, SyntaxKind::AttributeDict, bad)?;
        Ok(())
    }

    fn dictionary_entries(&mut self) -> Result<bool, CompactError> {
        let mut good = self.expect(TokenKind::LBrace)?;
        self.trivia()?;
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.position;
            let attr = self.builder.start();
            let mut bad = false;
            if self.at_identifier() || self.at(TokenKind::String) {
                let key_range = self.tokens[self.position].range();
                let key = std::str::from_utf8(
                    self.source
                        .get(key_range.start() as usize..key_range.end() as usize)
                        .unwrap_or_default(),
                )
                .unwrap_or("");
                self.bump()?;
                self.trivia()?;
                if key == "no_inline" && (self.at(TokenKind::RBrace) || self.at(TokenKind::Comma)) {
                    // The registered func.func schema admits MLIR's unit
                    // attribute spelling: {no_inline}.
                } else {
                    bad |= !self.expect(TokenKind::Equal)?;
                    self.trivia()?;
                    if self.at(TokenKind::RBrace) {
                        self.diagnostic();
                        bad = true;
                    } else {
                        bad |= !self.attribute_value()?;
                    }
                }
            } else {
                bad = true;
                self.error_token()?;
            }
            self.builder
                .complete_with_error(attr, SyntaxKind::Attribute, bad)?;
            good &= !bad;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else if !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                good = false;
                self.diagnostic();
            }
            self.ensure_progress(before)?;
        }
        good &= self.expect(TokenKind::RBrace)?;
        Ok(good)
    }

    fn function_type(&mut self) -> Result<(), CompactError> {
        let ty = self.builder.start();
        self.type_list(0)?;
        self.trivia()?;
        self.expect(TokenKind::Arrow)?;
        self.trivia()?;
        if self.at_type_start() {
            self.type_syntax(0)?;
        } else {
            self.type_list(0)?;
        }
        self.builder.complete(ty, SyntaxKind::FunctionType)?;
        Ok(())
    }

    fn type_list(&mut self, depth: usize) -> Result<(), CompactError> {
        self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while self.at_type_start() {
            self.type_syntax(depth)?;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        self.expect(TokenKind::RParen)?;
        Ok(())
    }

    fn at_type_start(&self) -> bool {
        matches!(
            self.current(),
            TokenKind::IntType
                | TokenKind::FloatType
                | TokenKind::IndexType
                | TokenKind::ExclamationIdentifier
                | TokenKind::Tuple
                | TokenKind::Tensor
                | TokenKind::Vector
                | TokenKind::MemRef
                | TokenKind::AffineMap
                | TokenKind::AffineSet
        )
    }

    fn type_syntax(&mut self, depth: usize) -> Result<bool, CompactError> {
        if depth >= MAX_TYPE_DEPTH {
            self.diagnostic();
            self.recover_type_boundary()?;
            return Ok(false);
        }
        let kind = match self.current() {
            TokenKind::IntType => SyntaxKind::IntegerType,
            TokenKind::FloatType => SyntaxKind::FloatType,
            TokenKind::IndexType => SyntaxKind::IndexType,
            TokenKind::ExclamationIdentifier if self.nth_nontrivia(1) == Some(TokenKind::Less) => {
                return self.opaque(SyntaxKind::OpaqueType);
            }
            TokenKind::ExclamationIdentifier => SyntaxKind::TypeAlias,
            TokenKind::Tuple => return self.tuple_type(depth + 1),
            TokenKind::Tensor => return self.shaped_type(SyntaxKind::TensorType, depth + 1),
            TokenKind::Vector => return self.shaped_type(SyntaxKind::VectorType, depth + 1),
            TokenKind::MemRef => return self.shaped_type(SyntaxKind::MemRefType, depth + 1),
            TokenKind::AffineMap => return self.affine_value(SyntaxKind::AffineMap),
            TokenKind::AffineSet => return self.affine_value(SyntaxKind::IntegerSet),
            _ => return Ok(false),
        };
        let marker = self.builder.start();
        self.bump()?;
        self.builder.complete(marker, kind)?;
        Ok(true)
    }

    fn tuple_type(&mut self, depth: usize) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        while self.at_type_start() {
            good &= self.type_syntax(depth)?;
            self.trivia()?;
            if self.at(TokenKind::LBrace) {
                self.attribute_dict()?;
                self.trivia()?;
            }
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump()?;
            self.trivia()?;
        }
        good &= self.expect(TokenKind::Greater)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::TupleType, !good)?;
        Ok(good)
    }

    fn shaped_type(&mut self, kind: SyntaxKind, depth: usize) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        while matches!(
            self.current(),
            TokenKind::Integer | TokenKind::Question | TokenKind::Star
        ) || (kind == SyntaxKind::VectorType && self.at(TokenKind::LBracket))
        {
            let dimension = self.builder.start();
            let mut dimension_good = true;
            if self.at(TokenKind::LBracket) {
                self.bump()?;
                self.trivia()?;
                dimension_good &= self.expect(TokenKind::Integer)?;
                self.trivia()?;
                dimension_good &= self.expect(TokenKind::RBracket)?;
            } else {
                self.bump()?;
            }
            self.builder.complete_with_error(
                dimension,
                SyntaxKind::ShapedDimension,
                !dimension_good,
            )?;
            good &= dimension_good;
            self.trivia()?;
            good &= self.expect(TokenKind::X)?;
            self.trivia()?;
        }
        if self.at_type_start() {
            good &= self.type_syntax(depth)?;
        } else {
            good = false;
            self.diagnostic();
            self.recover_type_boundary()?;
        }
        self.trivia()?;
        if kind == SyntaxKind::TensorType && self.at(TokenKind::Comma) {
            let encoding = self.builder.start();
            self.bump()?;
            self.trivia()?;
            let encoding_good = self.attribute_value()?;
            self.builder.complete_with_error(
                encoding,
                SyntaxKind::TensorEncoding,
                !encoding_good,
            )?;
            good &= encoding_good;
            self.trivia()?;
        } else if kind == SyntaxKind::MemRefType && self.at(TokenKind::Comma) {
            good &= self.memref_suffix()?;
        }
        good &= self.expect(TokenKind::Greater)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn memref_suffix(&mut self) -> Result<bool, CompactError> {
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Integer) {
            good &= self.memref_memory_space()?;
            return Ok(good);
        }

        let layout = self.builder.start();
        let layout_good = match self.current() {
            TokenKind::AffineMap => self.affine_value(SyntaxKind::AffineMap)?,
            TokenKind::Strided => self.balanced_angle_node(SyntaxKind::StridedLayout)?,
            _ => self.attribute_value()?,
        };
        self.builder
            .complete_with_error(layout, SyntaxKind::MemRefLayout, !layout_good)?;
        good &= layout_good;
        self.trivia()?;
        if self.at(TokenKind::Comma) {
            self.bump()?;
            self.trivia()?;
            good &= self.memref_memory_space()?;
        }
        Ok(good)
    }

    fn memref_memory_space(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        let good = if self.at(TokenKind::Integer) {
            self.bump()?;
            true
        } else {
            self.attribute_value()?
        };
        self.builder
            .complete_with_error(marker, SyntaxKind::MemRefMemorySpace, !good)?;
        self.trivia()?;
        Ok(good)
    }

    fn balanced_angle_node(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        let mut stack = vec![TokenKind::Greater];
        while let Some(expected) = stack.last().copied() {
            if self.at(TokenKind::Eof) || self.at(TokenKind::RBrace) {
                good = false;
                self.diagnostic();
                break;
            }
            let current = self.current();
            if current == expected {
                stack.pop();
            } else if let Some(close) = close_for(current) {
                if stack.len() >= MAX_TYPE_DEPTH {
                    good = false;
                    self.diagnostic();
                    break;
                }
                stack.push(close);
            }
            self.bump()?;
        }
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn affine_value(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::Less)?;
        self.trivia()?;
        good &= self.affine_identifier_list()?;
        self.trivia()?;
        if self.at(TokenKind::LBracket) {
            good &= self.affine_identifier_list()?;
            self.trivia()?;
        }
        good &= self.expect(if kind == SyntaxKind::AffineMap {
            TokenKind::Arrow
        } else {
            TokenKind::Colon
        })?;
        self.trivia()?;
        good &= self.expect(TokenKind::LParen)?;
        self.trivia()?;
        while !self.at(TokenKind::RParen)
            && !self.at(TokenKind::Greater)
            && !self.at(TokenKind::Eof)
        {
            let before = self.position;
            let expression = self.affine_expression(0, 0)?;
            good &= expression.is_some();
            self.trivia()?;
            if kind == SyntaxKind::IntegerSet {
                let constraint = expression
                    .map(|expr| self.builder.precede(expr))
                    .transpose()?;
                if matches!(
                    self.current(),
                    TokenKind::Less | TokenKind::Greater | TokenKind::Equal
                ) {
                    self.bump()?;
                    if self.at(TokenKind::Equal) {
                        self.bump()?;
                    } else {
                        good = false;
                        self.diagnostic();
                    }
                    self.trivia()?;
                    good &= self.affine_expression(0, 0)?.is_some();
                } else {
                    good = false;
                    self.diagnostic();
                }
                if let Some(constraint) = constraint {
                    self.builder.complete_with_error(
                        constraint,
                        SyntaxKind::AffineConstraint,
                        !good,
                    )?;
                }
            }
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else {
                break;
            }
            self.ensure_progress(before)?;
        }
        good &= self.expect(TokenKind::RParen)?;
        self.trivia()?;
        good &= self.expect(TokenKind::Greater)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn affine_identifier_list(&mut self) -> Result<bool, CompactError> {
        let close = if self.at(TokenKind::LParen) {
            TokenKind::RParen
        } else {
            TokenKind::RBracket
        };
        let mut good = self.expect(if close == TokenKind::RParen {
            TokenKind::LParen
        } else {
            TokenKind::LBracket
        })?;
        self.trivia()?;
        while self.at_identifier() {
            self.bump()?;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else {
                break;
            }
        }
        good &= self.expect(close)?;
        Ok(good)
    }

    fn affine_expression(
        &mut self,
        _min_precedence: u8,
        _depth: usize,
    ) -> Result<Option<CompletedMarker>, CompactError> {
        let mut operands = Vec::<CompletedMarker>::new();
        let mut operators = Vec::<u8>::new();
        let mut delimiters = Vec::<(usize, usize, Marker, Option<Marker>)>::new();
        let mut expect_operand = true;
        let mut bad = false;

        loop {
            self.trivia()?;
            if expect_operand {
                if self.at(TokenKind::Minus) {
                    let unary = self.builder.start();
                    self.bump()?;
                    self.trivia()?;
                    if self.at(TokenKind::LParen) {
                        if delimiters.len() >= MAX_TYPE_DEPTH {
                            self.diagnostic();
                            bad = true;
                            operands.push(self.builder.complete_with_error(
                                unary,
                                SyntaxKind::AffineExpression,
                                true,
                            )?);
                            break;
                        }
                        let marker = self.builder.start();
                        self.bump()?;
                        delimiters.push((operators.len(), operands.len(), marker, Some(unary)));
                        continue;
                    }
                    let good = self.at(TokenKind::Integer) || self.at_identifier();
                    if good {
                        self.bump()?;
                    } else {
                        self.diagnostic();
                        bad = true;
                    }
                    operands.push(self.builder.complete_with_error(
                        unary,
                        SyntaxKind::AffineExpression,
                        !good,
                    )?);
                    expect_operand = false;
                    continue;
                } else if self.at(TokenKind::LParen) {
                    if delimiters.len() >= MAX_TYPE_DEPTH {
                        self.diagnostic();
                        bad = true;
                        break;
                    }
                    let marker = self.builder.start();
                    self.bump()?;
                    delimiters.push((operators.len(), operands.len(), marker, None));
                    continue;
                }
                if self.at(TokenKind::Integer) || self.at_identifier() {
                    let marker = self.builder.start();
                    let good = self.at(TokenKind::Integer) || self.at_identifier();
                    if good {
                        self.bump()?;
                    } else {
                        self.diagnostic();
                        bad = true;
                    }
                    operands.push(self.builder.complete_with_error(
                        marker,
                        SyntaxKind::AffineExpression,
                        !good,
                    )?);
                    expect_operand = false;
                    continue;
                }
                self.diagnostic();
                bad = true;
                break;
            }

            let precedence = match self.current() {
                TokenKind::Plus | TokenKind::Minus => Some(1),
                TokenKind::Star | TokenKind::Mod | TokenKind::FloorDiv | TokenKind::CeilDiv => {
                    Some(2)
                }
                _ => None,
            };
            if let Some(precedence) = precedence {
                let floor = delimiters.last().map_or(0, |frame| frame.0);
                while operators.len() > floor
                    && operators.last().is_some_and(|top| *top >= precedence)
                {
                    bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
                }
                self.bump()?;
                operators.push(precedence);
                expect_operand = true;
                continue;
            }

            if self.at(TokenKind::RParen) && !delimiters.is_empty() {
                let (operator_base, operand_base, marker, unary) = delimiters.pop().unwrap();
                while operators.len() > operator_base {
                    bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
                }
                let inner_good = operands.len() == operand_base + 1 && !expect_operand;
                self.bump()?;
                let inner = operands.pop();
                operands.truncate(operand_base);
                bad |= inner.is_none();
                let mut grouped = self.builder.complete_with_error(
                    marker,
                    SyntaxKind::AffineExpression,
                    !inner_good,
                )?;
                if let Some(unary) = unary {
                    grouped = self.builder.complete_with_error(
                        unary,
                        SyntaxKind::AffineExpression,
                        !inner_good,
                    )?;
                }
                operands.push(grouped);
                expect_operand = false;
                continue;
            }
            break;
        }

        if expect_operand && !operators.is_empty() {
            self.diagnostic();
            bad = true;
        }
        while let Some((operator_base, operand_base, marker, unary)) = delimiters.pop() {
            while operators.len() > operator_base {
                bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
            }
            let inner = operands.pop();
            operands.truncate(operand_base);
            bad |= inner.is_none();
            let mut grouped =
                self.builder
                    .complete_with_error(marker, SyntaxKind::AffineExpression, true)?;
            if let Some(unary) = unary {
                grouped =
                    self.builder
                        .complete_with_error(unary, SyntaxKind::AffineExpression, true)?;
            }
            operands.push(grouped);
        }
        while !operators.is_empty() {
            bad |= !self.reduce_affine_operator(&mut operands, &mut operators)?;
        }
        let result = operands.pop();
        if !operands.is_empty() {
            self.diagnostic();
            bad = true;
        }
        if bad {
            if let Some(result) = result {
                let marker = self.builder.precede(result)?;
                return Ok(Some(self.builder.complete_with_error(
                    marker,
                    SyntaxKind::AffineExpression,
                    true,
                )?));
            }
        }
        Ok(result)
    }

    fn reduce_affine_operator(
        &mut self,
        operands: &mut Vec<CompletedMarker>,
        operators: &mut Vec<u8>,
    ) -> Result<bool, CompactError> {
        operators.pop();
        let Some(right) = operands.pop() else {
            return Ok(false);
        };
        let Some(left) = operands.pop() else {
            operands.push(right);
            return Ok(false);
        };
        let parent = self.builder.precede(left)?;
        operands.push(
            self.builder
                .complete(parent, SyntaxKind::AffineExpression)?,
        );
        Ok(true)
    }

    fn recover_type_boundary(&mut self) -> Result<(), CompactError> {
        while !matches!(
            self.current(),
            TokenKind::Comma
                | TokenKind::RParen
                | TokenKind::RBrace
                | TokenKind::Greater
                | TokenKind::Eof
        ) {
            self.error_token()?;
        }
        Ok(())
    }

    fn attribute_value(&mut self) -> Result<bool, CompactError> {
        match self.current() {
            TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Integer
            | TokenKind::WideInteger
            | TokenKind::Float => self.numeric_attribute(),
            TokenKind::String => self.string_attribute(),
            TokenKind::IntType
            | TokenKind::FloatType
            | TokenKind::IndexType
            | TokenKind::ExclamationIdentifier
            | TokenKind::Tuple
            | TokenKind::Tensor
            | TokenKind::Vector
            | TokenKind::MemRef
            | TokenKind::AffineMap
            | TokenKind::AffineSet => {
                let marker = self.builder.start();
                let good = self.type_syntax(0)?;
                self.builder
                    .complete_with_error(marker, SyntaxKind::TypeAttribute, !good)?;
                Ok(good)
            }
            TokenKind::Dense => self.payload_attribute(SyntaxKind::DenseElementsAttribute),
            TokenKind::Sparse => self.payload_attribute(SyntaxKind::SparseElementsAttribute),
            TokenKind::DenseResource => {
                self.payload_attribute(SyntaxKind::DenseResourceElementsAttribute)
            }
            TokenKind::HashIdentifier if self.nth_nontrivia(1) == Some(TokenKind::Less) => {
                self.opaque(SyntaxKind::OpaqueAttribute)
            }
            TokenKind::HashIdentifier => self.leaf(SyntaxKind::AttributeAlias),
            TokenKind::AtIdentifier => self.symbol_reference(),
            TokenKind::BareIdentifier if self.current_text() == "type" => {
                let marker = self.builder.start();
                self.bump()?;
                self.trivia()?;
                let mut good = self.expect(TokenKind::Less)?;
                self.trivia()?;
                if self.at(TokenKind::LParen) {
                    self.function_type()?;
                } else {
                    good &= self.type_syntax(0)?;
                }
                self.trivia()?;
                good &= self.expect(TokenKind::Greater)?;
                self.builder
                    .complete_with_error(marker, SyntaxKind::TypeAttribute, !good)?;
                Ok(good)
            }
            TokenKind::BareIdentifier if self.current_text() == "unit" => {
                let marker = self.builder.start();
                self.bump()?;
                self.builder.complete(marker, SyntaxKind::AttributeAlias)?;
                Ok(true)
            }
            TokenKind::BareIdentifier if matches!(self.current_text(), "true" | "false") => {
                self.leaf(SyntaxKind::BooleanAttribute)
            }
            TokenKind::LBracket => self.array_attribute(),
            TokenKind::LBrace => self.dictionary_attribute(),
            TokenKind::Loc => self.location_attribute(),
            _ => {
                self.error_token()?;
                Ok(false)
            }
        }
    }

    fn constant_value(&mut self) -> Result<bool, CompactError> {
        if self.at(TokenKind::String) {
            let marker = self.builder.start();
            self.bump()?;
            self.builder.complete(marker, SyntaxKind::StringAttribute)?;
            return Ok(true);
        }
        if matches!(
            self.current(),
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Integer
                | TokenKind::WideInteger
                | TokenKind::Float
        ) {
            let marker = self.builder.start();
            if matches!(self.current(), TokenKind::Plus | TokenKind::Minus) {
                self.bump()?;
            }
            let kind = match self.current() {
                TokenKind::Integer => SyntaxKind::IntegerAttribute,
                TokenKind::WideInteger => SyntaxKind::WideNumber,
                TokenKind::Float => SyntaxKind::FloatAttribute,
                _ => {
                    self.diagnostic();
                    self.builder
                        .complete_with_error(marker, SyntaxKind::IntegerAttribute, true)?;
                    return Ok(false);
                }
            };
            self.bump()?;
            self.builder.complete(marker, kind)?;
            Ok(true)
        } else {
            self.attribute_value()
        }
    }

    fn leaf(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.builder.complete(marker, kind)?;
        Ok(true)
    }

    fn numeric_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if matches!(self.current(), TokenKind::Plus | TokenKind::Minus) {
            self.bump()?;
        }
        let mut kind = match self.current() {
            TokenKind::Integer => SyntaxKind::IntegerAttribute,
            TokenKind::WideInteger => SyntaxKind::WideNumber,
            TokenKind::Float => SyntaxKind::FloatAttribute,
            _ => {
                self.diagnostic();
                self.builder
                    .complete_with_error(marker, SyntaxKind::IntegerAttribute, true)?;
                return Ok(false);
            }
        };
        let numeric_range = self.tokens[self.position].range();
        let oversized = self.current_token_bytes() > self.limits.max_numeric_literal_bytes;
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Colon) {
            self.bump()?;
            self.trivia()?;
            good = self.type_syntax(0)?;
            if self
                .tokens
                .get(self.position - 1)
                .is_some_and(|token| token.kind() == TokenKind::FloatType)
            {
                kind = SyntaxKind::FloatAttribute;
            }
            if !good {
                self.diagnostic();
            }
        }
        if oversized {
            self.diagnostics.push(ParseDiagnostic {
                range: numeric_range,
                kind: ParseDiagnosticKind::Syntax,
            });
            self.builder
                .complete_with_error(marker, SyntaxKind::WideNumber, true)?;
            return Ok(false);
        }
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn payload_attribute(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut good = self.expect(TokenKind::Less)?;
        let payload_start = self.tokens[self.position.saturating_sub(1)].range().end();
        good &= self.scan_balanced_payload(payload_start)?;
        self.trivia()?;
        good &= self.expect(TokenKind::Colon)?;
        self.trivia()?;
        good &= self.type_syntax(0)?;
        self.builder.complete_with_error(marker, kind, !good)?;
        Ok(good)
    }

    fn scan_balanced_payload(&mut self, payload_start: u32) -> Result<bool, CompactError> {
        let mut stack = vec![TokenKind::Greater];
        let mut good = true;
        while let Some(expected) = stack.last().copied() {
            if self.at(TokenKind::Eof) || (stack.len() == 1 && self.at(TokenKind::RBrace)) {
                self.diagnostic();
                return Ok(false);
            }
            let current = self.current();
            if current == expected {
                stack.pop();
                self.bump()?;
                continue;
            }
            if is_close(current) {
                self.diagnostic();
                good = false;
                if current == TokenKind::RBrace {
                    return Ok(false);
                }
                self.bump()?;
                continue;
            }
            if let Some(close) = close_for(current) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic();
                    good = false;
                } else {
                    stack.push(close);
                }
            }
            if matches!(
                current,
                TokenKind::Integer | TokenKind::WideInteger | TokenKind::Float
            ) && self.current_token_bytes() > self.limits.max_numeric_literal_bytes
            {
                self.diagnostic();
                good = false;
            }
            if self.tokens[self.position]
                .range()
                .end()
                .saturating_sub(payload_start) as usize
                > self.limits.max_payload_bytes
            {
                self.diagnostic();
                good = false;
            }
            self.bump()?;
        }
        Ok(good)
    }

    fn string_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = true;
        if self.at(TokenKind::Colon) {
            self.bump()?;
            self.trivia()?;
            good = self.type_syntax(0)?;
            if !good {
                self.diagnostic();
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::StringAttribute, !good)?;
        Ok(good)
    }

    fn symbol_reference(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut good = true;
        loop {
            self.trivia()?;
            if !self.at(TokenKind::Colon) {
                break;
            }
            self.bump()?;
            good &= self.expect(TokenKind::Colon)?;
            good &= self.expect(TokenKind::AtIdentifier)?;
            if !good {
                break;
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::SymbolReference, !good)?;
        Ok(good)
    }

    fn array_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBracket, TokenKind::RBracket)? {
            self.builder
                .complete_with_error(marker, SyntaxKind::ArrayAttribute, true)?;
            return Ok(false);
        }
        let mut good = self.expect(TokenKind::LBracket)?;
        self.trivia()?;
        while !self.at(TokenKind::RBracket)
            && !self.at(TokenKind::RBrace)
            && !self.at(TokenKind::Eof)
        {
            let before = self.position;
            good &= self.attribute_value()?;
            self.trivia()?;
            if self.at(TokenKind::Comma) {
                self.bump()?;
                self.trivia()?;
            } else if !self.at(TokenKind::RBracket) {
                good = false;
                self.diagnostic();
            }
            debug_assert!(self.position > before);
        }
        good &= self.expect(TokenKind::RBracket)?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(marker, SyntaxKind::ArrayAttribute, !good)?;
        Ok(good)
    }

    fn dictionary_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        if !self.enter_attribute_container(TokenKind::LBrace, TokenKind::RBrace)? {
            self.builder
                .complete_with_error(marker, SyntaxKind::DictionaryAttribute, true)?;
            return Ok(false);
        }
        let good = self.dictionary_entries()?;
        self.nesting_depth -= 1;
        self.builder
            .complete_with_error(marker, SyntaxKind::DictionaryAttribute, !good)?;
        Ok(good)
    }

    fn location_attribute(&mut self) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        self.trivia()?;
        let mut good = self.expect(TokenKind::LParen)?;
        self.trivia()?;
        let detail = self.builder.start();
        let detail_kind = match self.current() {
            TokenKind::Unknown => SyntaxKind::UnknownLocation,
            TokenKind::CallSite => SyntaxKind::CallSiteLocation,
            TokenKind::Fused => SyntaxKind::FusedLocation,
            TokenKind::String if self.nth_nontrivia(1) == Some(TokenKind::Colon) => {
                SyntaxKind::FileLineColLocation
            }
            _ => SyntaxKind::NameLocation,
        };
        let mut stack = vec![TokenKind::RParen];
        while !self.at(TokenKind::Eof) {
            let kind = self.current();
            let following_operation = stack.len() == 1
                && kind == TokenKind::String
                && self.nth_nontrivia(1) == Some(TokenKind::LParen)
                && !matches!(
                    self.nth_nontrivia(2),
                    Some(
                        TokenKind::HashIdentifier
                            | TokenKind::ExclamationIdentifier
                            | TokenKind::String
                            | TokenKind::Loc
                            | TokenKind::Unknown
                            | TokenKind::CallSite
                            | TokenKind::Fused
                    )
                );
            let at_enclosing_boundary = !stack.is_empty()
                && (kind == TokenKind::RBrace
                    || (kind == TokenKind::Comma && stack.last() == Some(&TokenKind::RParen)));
            if (stack.len() == 1 && kind == TokenKind::RParen)
                || at_enclosing_boundary
                || following_operation
            {
                break;
            }
            if let Some(close) = close_for(kind) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
                    good = false;
                } else {
                    stack.push(close);
                }
            } else if stack.last() == Some(&kind) {
                stack.pop();
            }
            self.bump()?;
        }
        if self.at(TokenKind::Eof)
            || self.at(TokenKind::RBrace)
            || (self.at(TokenKind::String)
                && self.nth_nontrivia(1) == Some(TokenKind::LParen)
                && !matches!(
                    self.nth_nontrivia(2),
                    Some(
                        TokenKind::HashIdentifier
                            | TokenKind::ExclamationIdentifier
                            | TokenKind::String
                            | TokenKind::Loc
                            | TokenKind::Unknown
                            | TokenKind::CallSite
                            | TokenKind::Fused
                    )
                ))
            || (self.at(TokenKind::Comma) && stack.last() == Some(&TokenKind::RParen))
        {
            good = false;
            self.diagnostic();
        }
        self.builder
            .complete_with_error(detail, detail_kind, !good)?;
        good &= self.expect(TokenKind::RParen)?;
        self.builder
            .complete_with_error(marker, SyntaxKind::LocationAttribute, !good)?;
        Ok(good)
    }

    fn opaque(&mut self, kind: SyntaxKind) -> Result<bool, CompactError> {
        let marker = self.builder.start();
        self.bump()?;
        let mut bad = false;
        if self.at(TokenKind::Less) {
            let body = self.builder.start();
            self.bump()?;
            let payload_start = self.tokens[self.position - 1].range().end();
            bad = !self.scan_balanced_payload(payload_start)?;
            self.builder.complete_with_error(
                body,
                if kind == SyntaxKind::OpaqueType {
                    SyntaxKind::OpaqueTypeBody
                } else {
                    SyntaxKind::OpaqueAttributeBody
                },
                bad,
            )?;
        }
        self.builder.complete_with_error(marker, kind, bad)?;
        Ok(!bad)
    }

    fn trivia(&mut self) -> Result<(), CompactError> {
        while matches!(
            self.current(),
            TokenKind::Whitespace | TokenKind::LineComment
        ) {
            self.bump()?;
        }
        Ok(())
    }
    fn expect(&mut self, kind: TokenKind) -> Result<bool, CompactError> {
        if self.at(kind) {
            self.bump()?;
            Ok(true)
        } else {
            self.diagnostic();
            Ok(false)
        }
    }
    fn error_token(&mut self) -> Result<(), CompactError> {
        let error = self.builder.start();
        self.diagnostic();
        if !self.at(TokenKind::Eof) {
            self.bump()?;
        }
        self.builder
            .complete_with_error(error, SyntaxKind::Error, true)?;
        Ok(())
    }
    fn diagnostic(&mut self) {
        self.diagnostic_kind(ParseDiagnosticKind::Syntax);
    }
    fn diagnostic_kind(&mut self, kind: ParseDiagnosticKind) {
        self.diagnostics.push(ParseDiagnostic {
            range: self.tokens[self.position].range(),
            kind,
        });
    }
    fn ensure_progress(&mut self, before: usize) -> Result<(), CompactError> {
        if self.position == before && !self.at(TokenKind::Eof) {
            self.diagnostic_kind(ParseDiagnosticKind::ProgressLimit);
            self.error_token()?;
        }
        Ok(())
    }
    fn unparsed_custom_operation(&mut self, marker: Option<Marker>) -> Result<(), CompactError> {
        let marker = marker.unwrap_or_else(|| self.builder.start());
        self.diagnostic_kind(ParseDiagnosticKind::UnknownCustomOperation);
        let start = self.position;
        let mut stack = Vec::new();
        let mut line_boundary = false;
        let mut completed_payload = false;
        while !self.at(TokenKind::Eof) {
            let current = self.current();
            if stack.is_empty() && current == TokenKind::LBrace && self.region_shaped_body() {
                self.region()?;
                completed_payload = true;
                continue;
            }
            if stack.is_empty()
                && (current == TokenKind::RBrace
                    || current == TokenKind::CaretIdentifier
                    || self.is_generic_operation_start()
                    || (self.position > start
                        && ((current == TokenKind::BareIdentifier
                            && self.bare_custom_operation_start()
                            && (line_boundary || self.current_text().contains('.')))
                            || (current == TokenKind::PercentIdentifier
                                && self.result_custom_operation_start()))
                        && (line_boundary || completed_payload))
                    || (self.position > start
                        && current == TokenKind::BareIdentifier
                        && self.previous_nontrivia() == Some(TokenKind::PercentIdentifier)
                        && self.nth_nontrivia(1) == Some(TokenKind::PercentIdentifier)))
            {
                break;
            }
            if stack.is_empty() && completed_payload && !is_trivia(current) {
                completed_payload = false;
            }
            if stack.last() == Some(&current) {
                stack.pop();
                if stack.is_empty() {
                    completed_payload = true;
                }
            } else if let Some(close) = close_for(current) {
                if stack.len() >= self.limits.max_delimiter_depth {
                    self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
                } else {
                    stack.push(close);
                }
            } else if is_close(current) && stack.is_empty() {
                break;
            }
            let token_has_line_break = current == TokenKind::Whitespace
                && self
                    .source
                    .get(
                        self.tokens[self.position].range().start() as usize
                            ..self.tokens[self.position].range().end() as usize,
                    )
                    .is_some_and(|bytes| bytes.contains(&b'\n'));
            self.bump()?;
            if token_has_line_break {
                line_boundary = true;
            } else if !is_trivia(current) {
                line_boundary = false;
            }
        }
        self.builder
            .complete_with_error(marker, SyntaxKind::UnparsedCustomOperation, true)?;
        Ok(())
    }
    fn result_custom_operation_start(&self) -> bool {
        self.result_assignment_starts_operation(true)
    }
    fn bare_custom_operation_start(&self) -> bool {
        let range = self.tokens[self.position].range();
        self.source
            .get(range.start() as usize..range.end() as usize)
            != Some(b"attributes")
    }
    fn region_shaped_body(&self) -> bool {
        let mut index = self.position + 1;
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        match self.tokens.get(index).map(|token| token.kind()) {
            Some(TokenKind::CaretIdentifier | TokenKind::PercentIdentifier) => true,
            Some(TokenKind::String) => {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::LParen)
            }
            Some(TokenKind::RBrace) => false,
            Some(TokenKind::BareIdentifier) => {
                let mnemonic = self.tokens[index].range();
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Equal)
                    && self
                        .source
                        .get(mnemonic.start() as usize..mnemonic.end() as usize)
                        .is_some_and(|text| text.contains(&b'.'))
            }
            _ => false,
        }
    }
    fn is_generic_operation_start(&self) -> bool {
        if self.at(TokenKind::String) {
            return self.nth_nontrivia(1) == Some(TokenKind::LParen);
        }
        self.result_assignment_starts_operation(false)
    }
    fn result_assignment_starts_operation(&self, allow_bare: bool) -> bool {
        if !self.at(TokenKind::PercentIdentifier) {
            return false;
        }
        let mut index = self.position;
        loop {
            if self.tokens.get(index).map(|token| token.kind())
                != Some(TokenKind::PercentIdentifier)
            {
                return false;
            }
            index += 1;
            while self
                .tokens
                .get(index)
                .is_some_and(|token| is_trivia(token.kind()))
            {
                index += 1;
            }
            if self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::Colon) {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                if self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Integer) {
                    return false;
                }
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
            }
            if self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::Comma) {
                index += 1;
                while self
                    .tokens
                    .get(index)
                    .is_some_and(|token| is_trivia(token.kind()))
                {
                    index += 1;
                }
                continue;
            }
            break;
        }
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        if self.tokens.get(index).map(|token| token.kind()) != Some(TokenKind::Equal) {
            return false;
        }
        index += 1;
        while self
            .tokens
            .get(index)
            .is_some_and(|token| is_trivia(token.kind()))
        {
            index += 1;
        }
        matches!(
            self.tokens.get(index).map(|token| token.kind()),
            Some(TokenKind::String)
        ) || (allow_bare
            && self.tokens.get(index).map(|token| token.kind()) == Some(TokenKind::BareIdentifier))
    }
    fn previous_nontrivia(&self) -> Option<TokenKind> {
        self.tokens[..self.position]
            .iter()
            .rev()
            .find(|token| !is_trivia(token.kind()))
            .map(|token| token.kind())
    }
    fn recover_balanced_region(&mut self) -> Result<(), CompactError> {
        let mut depth = 0usize;
        while !self.at(TokenKind::Eof) {
            match self.current() {
                TokenKind::LBrace => depth = depth.saturating_add(1),
                TokenKind::RBrace if depth <= 1 => {
                    self.bump()?;
                    return Ok(());
                }
                TokenKind::RBrace => depth -= 1,
                _ => {}
            }
            self.bump()?;
        }
        Ok(())
    }
    fn enter_attribute_container(
        &mut self,
        open: TokenKind,
        close: TokenKind,
    ) -> Result<bool, CompactError> {
        if self.nesting_depth < self.limits.max_delimiter_depth {
            self.nesting_depth += 1;
            return Ok(true);
        }
        self.diagnostic_kind(ParseDiagnosticKind::DepthLimit);
        let mut depth = 0usize;
        while !self.at(TokenKind::Eof) {
            let current = self.current();
            if current == open {
                depth += 1;
            } else if current == close {
                self.bump()?;
                if depth <= 1 {
                    break;
                }
                depth -= 1;
                continue;
            }
            self.bump()?;
        }
        Ok(false)
    }
    fn current_token_bytes(&self) -> usize {
        self.tokens[self.position].range().len() as usize
    }
    fn bump(&mut self) -> Result<(), CompactError> {
        self.builder.token(self.position)?;
        self.position += 1;
        Ok(())
    }
    fn current(&self) -> TokenKind {
        self.tokens[self.position].kind()
    }
    fn current_text(&self) -> &str {
        let range = self.tokens[self.position].range();
        std::str::from_utf8(
            self.source
                .get(range.start() as usize..range.end() as usize)
                .unwrap_or_default(),
        )
        .unwrap_or("")
    }
    fn at(&self, kind: TokenKind) -> bool {
        self.current() == kind
    }
    fn at_identifier(&self) -> bool {
        matches!(
            self.current(),
            TokenKind::BareIdentifier
                | TokenKind::IntType
                | TokenKind::FloatType
                | TokenKind::IndexType
                | TokenKind::Tuple
                | TokenKind::Tensor
                | TokenKind::Vector
                | TokenKind::MemRef
                | TokenKind::AffineMap
                | TokenKind::AffineSet
                | TokenKind::Mod
                | TokenKind::FloorDiv
                | TokenKind::CeilDiv
                | TokenKind::Strided
                | TokenKind::Loc
                | TokenKind::Unknown
                | TokenKind::CallSite
                | TokenKind::Fused
        )
    }
    fn nth_nontrivia(&self, n: usize) -> Option<TokenKind> {
        self.tokens[self.position..]
            .iter()
            .filter(|token| !matches!(token.kind(), TokenKind::Whitespace | TokenKind::LineComment))
            .nth(n)
            .map(|token| token.kind())
    }
}

fn close_for(kind: TokenKind) -> Option<TokenKind> {
    match kind {
        TokenKind::Less => Some(TokenKind::Greater),
        TokenKind::LParen => Some(TokenKind::RParen),
        TokenKind::LBrace => Some(TokenKind::RBrace),
        TokenKind::LBracket => Some(TokenKind::RBracket),
        _ => None,
    }
}

fn is_close(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Greater | TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket
    )
}

/// Parses a deliberately tiny balanced-brace fixture grammar.
pub fn parse_brace_fixture(lexed: &Lexed) -> Result<SyntaxTree, CompactError> {
    let mut builder = EventBuilder::new();
    let root = builder.start();
    let mut braces = Vec::<Marker>::new();
    for (i, token) in lexed.tokens().iter().enumerate() {
        match token.kind() {
            TokenKind::LBrace => {
                let marker = builder.start();
                builder.token(i)?;
                braces.push(marker)
            }
            TokenKind::RBrace => {
                if let Some(marker) = braces.pop() {
                    builder.token(i)?;
                    builder.complete(marker, SyntaxKind::Region)?;
                } else {
                    let error = builder.start();
                    builder.token(i)?;
                    builder.complete_with_error(error, SyntaxKind::Error, true)?;
                }
            }
            _ => builder.token(i)?,
        }
    }
    while let Some(marker) = braces.pop() {
        builder.complete_with_error(marker, SyntaxKind::Region, true)?;
    }
    builder.complete(root, SyntaxKind::File)?;
    builder.finish(lexed.tokens().to_vec())
}

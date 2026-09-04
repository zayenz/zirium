use super::*;

/// Compact lossless CST storage and parser diagnostics for one file.
#[derive(Debug)]
pub struct ParsedSyntax {
    pub(super) tree: std::sync::Arc<SyntaxTree>,
    pub(super) diagnostics: Vec<ParseDiagnostic>,
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

pub(super) fn is_trivia(kind: TokenKind) -> bool {
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

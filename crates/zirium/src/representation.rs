use crate::lexer::{Token, TokenKind};
use std::{fmt, sync::OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntaxKind {
    File,
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
    DenseArrayAttribute,
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
    FileMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Tombstone,
    Start {
        kind: SyntaxKind,
        forward_parent: Option<u32>,
    },
    Token(u32),
    Finish {
        local_error: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Marker(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedMarker(usize);

#[derive(Debug, Default)]
pub struct EventBuilder {
    events: Vec<Event>,
}

impl EventBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn start(&mut self) -> Marker {
        let p = self.events.len();
        self.events.push(Event::Tombstone);
        Marker(p)
    }
    pub fn token(&mut self, index: usize) -> Result<(), CompactError> {
        self.events.push(Event::Token(
            u32::try_from(index).map_err(|_| CompactError::RepresentationTooLarge)?,
        ));
        Ok(())
    }
    pub fn complete(
        &mut self,
        marker: Marker,
        kind: SyntaxKind,
    ) -> Result<CompletedMarker, CompactError> {
        self.complete_with_error(marker, kind, false)
    }
    pub fn complete_with_error(
        &mut self,
        marker: Marker,
        kind: SyntaxKind,
        local_error: bool,
    ) -> Result<CompletedMarker, CompactError> {
        match self.events.get_mut(marker.0) {
            Some(e @ Event::Tombstone) => {
                *e = Event::Start {
                    kind,
                    forward_parent: None,
                }
            }
            _ => return Err(CompactError::InvalidMarker),
        }
        self.events.push(Event::Finish { local_error });
        Ok(CompletedMarker(marker.0))
    }
    pub fn abandon(&mut self, marker: Marker) -> Result<(), CompactError> {
        if marker.0 + 1 == self.events.len()
            && matches!(self.events.get(marker.0), Some(Event::Tombstone))
        {
            self.events.pop();
            Ok(())
        } else {
            Err(CompactError::InvalidMarker)
        }
    }
    pub fn precede(&mut self, marker: CompletedMarker) -> Result<Marker, CompactError> {
        let p = self.events.len();
        let distance = u32::try_from(p.checked_sub(marker.0).ok_or(CompactError::InvalidMarker)?)
            .map_err(|_| CompactError::RepresentationTooLarge)?;
        match self.events.get_mut(marker.0) {
            Some(Event::Start { forward_parent, .. }) if forward_parent.is_none() => {
                *forward_parent = Some(distance)
            }
            _ => return Err(CompactError::InvalidMarker),
        }
        self.events.push(Event::Tombstone);
        Ok(Marker(p))
    }
    pub fn finish(self, tokens: Vec<Token>) -> Result<SyntaxTree, CompactError> {
        SyntaxTree::from_events(self.events, tokens)
    }
    #[cfg(test)]
    pub(crate) fn into_events(self) -> Vec<Event> {
        self.events
    }
    pub fn events(&self) -> &[Event] {
        &self.events
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(u32);
impl NodeId {
    pub fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug)]
struct FlatNode {
    kind: SyntaxKind,
    first_token: u32,
    token_count: u32,
    subtree_end: u32,
    local_error: bool,
    subtree_error: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntaxElement<'a> {
    Node(NodeId),
    Token { index: usize, token: &'a Token },
}

#[derive(Debug, PartialEq, Eq)]
pub enum CompactError {
    Empty,
    MultipleRoots,
    TokenOutsideNode,
    UnexpectedFinish,
    UnclosedNode,
    InvalidMarker,
    InvalidForwardParent,
    InvalidTokenIndex,
    DuplicateToken,
    InvalidTokenRange,
    TokensOutOfOrder,
    InvalidSubtree,
    InvalidRootCoverage,
    InvalidErrorPropagation,
    RepresentationTooLarge,
}
impl fmt::Display for CompactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self {
            Self::Empty => "event stream is empty",
            Self::MultipleRoots => "event stream has multiple roots",
            Self::TokenOutsideNode => "token appears outside a node",
            Self::UnexpectedFinish => "node finish has no matching start",
            Self::UnclosedNode => "node is not closed",
            Self::InvalidMarker => "marker is invalid",
            Self::InvalidForwardParent => "forward-parent link is invalid",
            Self::InvalidTokenIndex => "token index is invalid",
            Self::DuplicateToken => "token is referenced more than once",
            Self::InvalidTokenRange => "token range is invalid",
            Self::TokensOutOfOrder => "tokens are out of source order",
            Self::InvalidSubtree => "subtree range is invalid",
            Self::InvalidRootCoverage => "root does not cover the complete tree",
            Self::InvalidErrorPropagation => "syntax error flags are inconsistent",
            Self::RepresentationTooLarge => "tree exceeds representation limits",
        };
        write!(f, "invalid syntax tree: {detail}")
    }
}
impl std::error::Error for CompactError {}

#[derive(Clone, Debug)]
pub struct SyntaxTree {
    nodes: Vec<FlatNode>,
    tokens: Vec<Token>,
    parents: OnceLock<Vec<Option<NodeId>>>,
}

impl SyntaxTree {
    pub fn from_events(events: Vec<Event>, tokens: Vec<Token>) -> Result<Self, CompactError> {
        let tree = Self::compact_events(events, tokens)?;
        tree.verify()?;
        Ok(tree)
    }
    #[cfg(test)]
    pub(crate) fn from_events_unverified(
        events: Vec<Event>,
        tokens: Vec<Token>,
    ) -> Result<Self, CompactError> {
        Self::compact_events(events, tokens)
    }
    fn compact_events(mut events: Vec<Event>, tokens: Vec<Token>) -> Result<Self, CompactError> {
        let mut nodes = Vec::<FlatNode>::new();
        let mut stored = Vec::with_capacity(tokens.len());
        let mut stack = Vec::new();
        let mut seen = vec![false; tokens.len()];
        let mut root = false;
        let mut compact = |event| -> Result<(), CompactError> {
            match event {
                Event::Tombstone => {}
                Event::Start {
                    kind,
                    forward_parent: None,
                } => {
                    if stack.is_empty() {
                        if root {
                            return Err(CompactError::MultipleRoots);
                        }
                        root = true;
                    }
                    let first_token = u32::try_from(stored.len())
                        .map_err(|_| CompactError::RepresentationTooLarge)?;
                    nodes.push(FlatNode {
                        kind,
                        first_token,
                        token_count: 0,
                        subtree_end: 0,
                        local_error: false,
                        subtree_error: false,
                    });
                    stack.push((nodes.len() - 1, false));
                }
                Event::Start { .. } => return Err(CompactError::InvalidForwardParent),
                Event::Token(i) => {
                    if stack.is_empty() {
                        return Err(CompactError::TokenOutsideNode);
                    }
                    let i = i as usize;
                    let token = *tokens.get(i).ok_or(CompactError::InvalidTokenIndex)?;
                    if std::mem::replace(&mut seen[i], true) {
                        return Err(CompactError::DuplicateToken);
                    }
                    if stored
                        .last()
                        .is_some_and(|p: &Token| token.range().start() < p.range().end())
                    {
                        return Err(CompactError::TokensOutOfOrder);
                    }
                    stored.push(token);
                }
                Event::Finish { local_error } => {
                    let (i, descendant_error) =
                        stack.pop().ok_or(CompactError::UnexpectedFinish)?;
                    nodes[i].token_count =
                        u32::try_from(stored.len() - nodes[i].first_token as usize)
                            .map_err(|_| CompactError::RepresentationTooLarge)?;
                    nodes[i].subtree_end = u32::try_from(nodes.len())
                        .map_err(|_| CompactError::RepresentationTooLarge)?;
                    nodes[i].local_error = local_error;
                    nodes[i].subtree_error = local_error || descendant_error;
                    if let Some((_, parent_descendant_error)) = stack.last_mut() {
                        *parent_descendant_error |= nodes[i].subtree_error;
                    }
                }
            }
            Ok(())
        };
        let mut chain = Vec::new();
        for p in 0..events.len() {
            chain.clear();
            let mut current = p;
            while let Event::Start {
                kind,
                forward_parent,
            } = *events
                .get(current)
                .ok_or(CompactError::InvalidForwardParent)?
            {
                events[current] = Event::Tombstone;
                chain.push(Event::Start {
                    kind,
                    forward_parent: None,
                });
                let Some(distance) = forward_parent else {
                    break;
                };
                current = current
                    .checked_add(distance as usize)
                    .filter(|n| *n > current && *n < events.len())
                    .ok_or(CompactError::InvalidForwardParent)?;
            }
            for event in chain.drain(..).rev() {
                compact(event)?;
            }
            if !matches!(events[p], Event::Tombstone) {
                compact(events[p])?;
            }
        }
        if !stack.is_empty() {
            return Err(CompactError::UnclosedNode);
        }
        if !root {
            return Err(CompactError::Empty);
        }
        if seen.iter().any(|s| !s) {
            return Err(CompactError::InvalidRootCoverage);
        }
        Ok(Self {
            nodes,
            tokens: stored,
            parents: OnceLock::new(),
        })
    }
    pub fn verify(&self) -> Result<(), CompactError> {
        let Some(root) = self.nodes.first() else {
            return Err(CompactError::Empty);
        };
        if root.subtree_end as usize != self.nodes.len()
            || root.first_token != 0
            || root.token_count as usize != self.tokens.len()
        {
            return Err(CompactError::InvalidRootCoverage);
        }
        if self
            .tokens
            .windows(2)
            .any(|p| p[1].range().start() < p[0].range().end())
        {
            return Err(CompactError::TokensOutOfOrder);
        }
        if self
            .nodes
            .iter()
            .any(|n| n.first_token as usize + n.token_count as usize > self.tokens.len())
        {
            return Err(CompactError::InvalidTokenRange);
        }
        for (i, n) in self.nodes.iter().enumerate() {
            let token_end = n.first_token as usize + n.token_count as usize;
            let end = n.subtree_end as usize;
            if end <= i || end > self.nodes.len() {
                return Err(CompactError::InvalidSubtree);
            }
            let mut error = n.local_error;
            let mut child = i + 1;
            let mut previous = n.first_token as usize;
            while child < end {
                let c = &self.nodes[child];
                let cend = c.first_token as usize + c.token_count as usize;
                if c.subtree_end as usize > end
                    || (c.first_token as usize) < previous
                    || cend > token_end
                {
                    return Err(CompactError::InvalidSubtree);
                }
                previous = cend;
                error |= c.subtree_error;
                child = c.subtree_end as usize;
            }
            if child != end {
                return Err(CompactError::InvalidSubtree);
            }
            if n.subtree_error != error {
                return Err(CompactError::InvalidErrorPropagation);
            }
        }
        Ok(())
    }
    pub fn root(&self) -> NodeId {
        NodeId(0)
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
    pub fn token(&self, index: usize) -> Option<&Token> {
        self.tokens.get(index)
    }
    pub fn node(&self, index: usize) -> Option<NodeId> {
        (index < self.nodes.len()).then_some(NodeId(index as u32))
    }
    pub fn token_indices(&self, n: NodeId) -> Option<std::ops::Range<usize>> {
        let n = self.nodes.get(n.index())?;
        let start = n.first_token as usize;
        Some(start..start + n.token_count as usize)
    }
    pub fn kind(&self, n: NodeId) -> Option<SyntaxKind> {
        self.nodes.get(n.index()).map(|n| n.kind)
    }
    pub fn subtree_end(&self, n: NodeId) -> Option<usize> {
        self.nodes.get(n.index()).map(|n| n.subtree_end as usize)
    }
    pub fn token_kind(&self, index: usize) -> Option<TokenKind> {
        self.tokens.get(index).map(|token| token.kind())
    }
    pub fn has_local_error(&self, n: NodeId) -> Option<bool> {
        self.nodes.get(n.index()).map(|n| n.local_error)
    }
    pub fn has_error(&self, n: NodeId) -> Option<bool> {
        self.nodes.get(n.index()).map(|n| n.subtree_error)
    }
    pub fn tokens(&self, n: NodeId) -> Option<&[Token]> {
        let n = self.nodes.get(n.index())?;
        self.tokens
            .get(n.first_token as usize..n.first_token as usize + n.token_count as usize)
    }
    pub fn text_range(&self, n: NodeId) -> Option<crate::source::TextRange> {
        let tokens = self.tokens(n)?;
        let first = tokens.first()?.range();
        let last = tokens.last()?.range();
        crate::source::TextRange::new(first.start(), last.end())
    }
    pub fn subtree(&self, n: NodeId) -> Option<impl Iterator<Item = NodeId> + '_> {
        let end = self.nodes.get(n.index())?.subtree_end as usize;
        Some((n.index()..end).map(|i| NodeId(i as u32)))
    }
    pub fn children(&self, n: NodeId) -> Option<impl Iterator<Item = NodeId> + '_> {
        let start = n.index() + 1;
        let end = self.nodes.get(n.index())?.subtree_end as usize;
        Some(
            std::iter::successors((start < end).then_some(start), move |i| {
                let next = self.nodes[*i].subtree_end as usize;
                (next < end).then_some(next)
            })
            .map(|i| NodeId(i as u32)),
        )
    }
    pub fn elements(&self, n: NodeId) -> Option<Vec<SyntaxElement<'_>>> {
        let record = self.nodes.get(n.index())?;
        let mut token = record.first_token as usize;
        let end = token + record.token_count as usize;
        let mut out = Vec::new();
        for child in self.children(n)? {
            let c = &self.nodes[child.index()];
            while token < c.first_token as usize {
                out.push(SyntaxElement::Token {
                    index: token,
                    token: &self.tokens[token],
                });
                token += 1
            }
            out.push(SyntaxElement::Node(child));
            token = c.first_token as usize + c.token_count as usize
        }
        while token < end {
            out.push(SyntaxElement::Token {
                index: token,
                token: &self.tokens[token],
            });
            token += 1
        }
        Some(out)
    }
    pub fn parent(&self, n: NodeId) -> Option<NodeId> {
        self.parent_index().get(n.index()).copied().flatten()
    }
    pub fn parent_index_is_built(&self) -> bool {
        self.parents.get().is_some()
    }
    pub fn build_parent_index(&self) {
        let _ = self.parent_index();
    }
    pub fn exact_retained_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<FlatNode>()
            + self.tokens.capacity() * size_of::<Token>()
            + self
                .parents
                .get()
                .map_or(0, |p| p.capacity() * size_of::<Option<NodeId>>())
    }
    fn parent_index(&self) -> &Vec<Option<NodeId>> {
        self.parents.get_or_init(|| {
            let mut parents = vec![None; self.nodes.len()];
            let mut ancestors = Vec::<(NodeId, usize)>::new();
            for (i, parent) in parents.iter_mut().enumerate() {
                while ancestors.last().is_some_and(|(_, e)| i >= *e) {
                    ancestors.pop();
                }
                *parent = ancestors.last().map(|(id, _)| *id);
                ancestors.push((NodeId(i as u32), self.nodes[i].subtree_end as usize));
            }
            parents
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer::TokenKind, source::TextRange};

    fn node(end: u32, first: u32, count: u32, local: bool, error: bool) -> FlatNode {
        FlatNode {
            kind: SyntaxKind::File,
            first_token: first,
            token_count: count,
            subtree_end: end,
            local_error: local,
            subtree_error: error,
        }
    }
    fn token(start: u32, end: u32) -> Token {
        Token::new(
            TokenKind::BareIdentifier,
            TextRange::new(start, end).unwrap(),
        )
    }
    fn raw(nodes: Vec<FlatNode>, tokens: Vec<Token>) -> SyntaxTree {
        SyntaxTree {
            nodes,
            tokens,
            parents: OnceLock::new(),
        }
    }
    #[test]
    fn rejects_malformed_trees() {
        assert_eq!(
            raw(vec![node(2, 0, 0, false, false)], vec![]).verify(),
            Err(CompactError::InvalidRootCoverage)
        );
        assert_eq!(
            raw(vec![node(1, 1, 0, false, false)], vec![]).verify(),
            Err(CompactError::InvalidRootCoverage)
        );
        assert_eq!(
            raw(vec![node(1, 0, 0, true, false)], vec![]).verify(),
            Err(CompactError::InvalidErrorPropagation)
        );
        assert_eq!(
            raw(
                vec![node(2, 0, 0, false, false), node(3, 0, 0, false, false)],
                vec![]
            )
            .verify(),
            Err(CompactError::InvalidSubtree)
        );
        assert_eq!(
            raw(
                vec![node(2, 0, 1, false, false), node(2, 1, 1, false, false),],
                vec![token(0, 1)],
            )
            .verify(),
            Err(CompactError::InvalidTokenRange)
        );
        assert_eq!(
            raw(
                vec![node(1, 0, 2, false, false)],
                vec![token(1, 2), token(0, 1)],
            )
            .verify(),
            Err(CompactError::TokensOutOfOrder)
        );
    }

    #[test]
    fn rejects_invalid_event_streams() {
        let finish = Event::Finish { local_error: false };
        let start = Event::Start {
            kind: SyntaxKind::File,
            forward_parent: None,
        };
        assert_eq!(
            SyntaxTree::from_events(vec![finish], vec![]).unwrap_err(),
            CompactError::UnexpectedFinish
        );
        assert_eq!(
            SyntaxTree::from_events(vec![start], vec![]).unwrap_err(),
            CompactError::UnclosedNode
        );
        assert_eq!(
            SyntaxTree::from_events(vec![Event::Token(0)], vec![token(0, 1)]).unwrap_err(),
            CompactError::TokenOutsideNode
        );
        assert_eq!(
            SyntaxTree::from_events(vec![start, Event::Token(1), finish], vec![token(0, 1)])
                .unwrap_err(),
            CompactError::InvalidTokenIndex
        );
        assert_eq!(
            SyntaxTree::from_events(
                vec![start, Event::Token(0), Event::Token(0), finish],
                vec![token(0, 1)],
            )
            .unwrap_err(),
            CompactError::DuplicateToken
        );
    }
}

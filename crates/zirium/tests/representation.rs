use zirium::{
    CompactError, EventBuilder, SyntaxElement, SyntaxKind,
    lexer::{TokenKind, lex},
    source::Source,
};

#[test]
fn markers_complete_abandon_and_nested_precede() {
    let source = Source::new(&b"abc"[..]).unwrap();
    let tokens = lex(&source).tokens().to_vec();
    let mut b = EventBuilder::new();
    let abandoned = b.start();
    b.abandon(abandoned).unwrap();
    let leaf = b.start();
    b.token(0).unwrap();
    let leaf = b.complete(leaf, SyntaxKind::Operation).unwrap();
    let parent = b.precede(leaf).unwrap();
    let parent = b.complete(parent, SyntaxKind::Region).unwrap();
    let root = b.precede(parent).unwrap();
    b.token(1).unwrap();
    b.complete(root, SyntaxKind::File).unwrap();
    let tree = b.finish(tokens).unwrap();
    let nodes: Vec<_> = tree.subtree(tree.root()).unwrap().collect();
    assert_eq!(nodes.len(), 3);
    assert_eq!(tree.kind(nodes[0]), Some(SyntaxKind::File));
    assert_eq!(tree.kind(nodes[1]), Some(SyntaxKind::Region));
    assert_eq!(tree.kind(nodes[2]), Some(SyntaxKind::Operation));
}

#[test]
fn rejects_invalid_markers() {
    let mut b = EventBuilder::new();
    let marker = b.start();
    b.token(0).unwrap();
    assert_eq!(b.abandon(marker), Err(CompactError::InvalidMarker));
    let done = b.complete(marker, SyntaxKind::File).unwrap();
    let _ = b.precede(done).unwrap();
    assert_eq!(b.precede(done), Err(CompactError::InvalidMarker));
}

#[test]
fn mixed_traversal_is_lossless_without_token_nodes() {
    let source = Source::new(&b"a { b } c"[..]).unwrap();
    let tree = zirium::parser::parse_brace_fixture(&lex(&source)).unwrap();
    fn append(tree: &zirium::SyntaxTree, source: &Source, node: zirium::NodeId, out: &mut Vec<u8>) {
        for element in tree.elements(node).unwrap() {
            match element {
                SyntaxElement::Node(child) => append(tree, source, child, out),
                SyntaxElement::Token { token, .. } if token.kind() != TokenKind::Eof => {
                    out.extend_from_slice(source.slice(token.range()).unwrap())
                }
                _ => {}
            }
        }
    }
    let mut bytes = vec![];
    append(&tree, &source, tree.root(), &mut bytes);
    assert_eq!(bytes, source.bytes());
    assert!(tree.node_count() < tree.token_count());
    assert_eq!(
        tree.subtree(tree.root()).unwrap().count(),
        tree.node_count()
    );
}

#[test]
fn parent_lookup_is_lazy() {
    let source = Source::new(&b"{{}}"[..]).unwrap();
    let tree = zirium::parser::parse_brace_fixture(&lex(&source)).unwrap();
    let nodes: Vec<_> = tree.subtree(tree.root()).unwrap().collect();
    let _ = tree.elements(tree.root()).unwrap();
    assert!(!tree.parent_index_is_built());
    assert_eq!(tree.parent(nodes[2]), Some(nodes[1]));
    assert!(tree.parent_index_is_built());
}

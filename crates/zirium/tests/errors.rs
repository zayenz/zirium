use std::error::Error;

use zirium::{
    parser::{ApplyTextEditsError, ParseFileError, ResourceLimitError, TextEditError},
    printer::{PreserveError, PrintError},
    semantic::{EditError, SemanticVerificationError, ValidationError},
    source::TextRange,
};

#[test]
fn parser_errors_have_human_messages_and_sources() {
    let range = TextRange::new(3, 5).unwrap();
    let cause = TextEditError::Overlapping(range);
    assert_eq!(
        cause.to_string(),
        "text edit range 3..5 overlaps another edit"
    );

    let wrapper = ApplyTextEditsError::Edit(cause);
    assert!(wrapper.to_string().starts_with("invalid text edit:"));
    assert!(wrapper.source().is_some());

    let parse = ParseFileError::ResourceLimit(ResourceLimitError {
        limit: 4,
        actual: 9,
    });
    assert_eq!(parse.to_string(), "file size 9 exceeds limit 4");
    assert_eq!(
        parse.source().unwrap().to_string(),
        "file size 9 exceeds limit 4"
    );
}

#[test]
fn semantic_and_output_wrappers_expose_their_causes() {
    let semantic = SemanticVerificationError::Structural(ValidationError::InvalidRetention);
    assert_eq!(
        semantic.to_string(),
        "structural verification failed: invalid semantic document: retained source, syntax, or mappings are inconsistent"
    );
    assert!(semantic.source().is_some());

    let edit = EditError::Semantic(semantic);
    assert!(
        edit.to_string()
            .contains("edited document is semantically invalid")
    );
    assert!(edit.source().is_some());

    let print = PrintError::InvalidDocument(ValidationError::InvalidList);
    assert_eq!(
        print.to_string(),
        "cannot print invalid document: invalid semantic document: an arena list reference is invalid"
    );
    assert!(print.source().is_some());

    let preserve = PreserveError::InvalidDocument(ValidationError::InvalidRetention);
    assert!(
        preserve
            .to_string()
            .starts_with("cannot preserve invalid document:")
    );
    assert!(preserve.source().is_some());

    let unknown = PreserveError::UnknownCustomSyntax(TextRange::new(8, 13).unwrap());
    assert_eq!(
        unknown.to_string(),
        "dirty replacement at 8..13 contains unknown custom syntax"
    );
}

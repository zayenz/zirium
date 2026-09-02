#![no_main]
use libfuzzer_sys::fuzz_target;
use zirium::lexer::lex;
use zirium::source::Source;

fuzz_target!(|data: &[u8]| {
    let source = Source::new(data.to_vec()).unwrap();
    let lexed = lex(&source);
    assert_eq!(lexed.reconstruct(&source), data);
    for token in lexed.tokens() {
        assert!(token.range().start() <= token.range().end());
        assert!(token.range().end() <= source.len());
    }
});

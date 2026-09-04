#![no_main]
use libfuzzer_sys::fuzz_target;
use zirium::parser::{ParseFileError, ParseLimits, ParsedFile};

fuzz_target!(|data: &[u8]| {
    match ParsedFile::parse_with_limits(data.to_vec(), ParseLimits {
        max_file_bytes: 4096,
        max_tokens: 256,
        max_delimiter_depth: 8,
        max_payload_bytes: 512,
        max_numeric_literal_bytes: 32,
        max_attribute_depth: 8,
        max_alias_expansion_depth: 8,
    }) {
        Ok(parsed) => {
            let tree = parsed.syntax().tree();
            let reconstructed = tree
                .tokens(tree.root())
                .unwrap()
                .iter()
                .flat_map(|token| parsed.source().slice(token.range()).unwrap())
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(reconstructed, data);
            tree.verify().unwrap();
        }
        Err(ParseFileError::ResourceLimit(_)) if data.len() > 4096 => {}
        Err(error) => panic!("unexpected parse failure: {error}"),
    }
});

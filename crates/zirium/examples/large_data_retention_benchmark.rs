use std::time::Instant;

use zirium::{
    parser::ParsedFile,
    semantic::{
        LoweringMode, RetentionProfile, SharedRegistry, lower_proving_fixture_with_retention,
    },
};

fn source(elements: usize) -> Vec<u8> {
    let mut dense = String::from("dense<[");
    for index in 0..elements {
        if index != 0 {
            dense.push_str(", ");
        }
        dense.push_str(&(index % 10).to_string());
    }
    dense.push_str("]> : tensor<");
    dense.push_str(&elements.to_string());
    dense.push_str("xi8>");

    let mut sparse_indices = String::from("[");
    let mut sparse_values = String::from("[");
    for index in 0..elements {
        if index != 0 {
            sparse_indices.push_str(", ");
            sparse_values.push_str(", ");
        }
        sparse_indices.push_str(&format!("[{}, 0]", index));
        sparse_values.push_str(&(index % 10).to_string());
    }
    sparse_indices.push(']');
    sparse_values.push(']');

    let opaque = "#vendor.attr<{ note = \"body\" }>";
    format!(
        "\"bench.payload\"() {{dense_value = {dense}, sparse_value = sparse<{sparse_indices}, {sparse_values}> : tensor<{elements}x1xi8>, resource_value = dense_resource<resource_handle> : tensor<1xi8>, wide_value = 0x1234567890abcdef : i128, opaque_value = {opaque} }} : () -> ()\n"
    )
    .into_bytes()
}

#[derive(Debug)]
struct Measurement {
    source_bytes: usize,
    semantic_attributes: usize,
    payload_bytes: usize,
    payload_blobs: usize,
    cst_nodes: usize,
    mappings: usize,
    direct_owned_bytes: usize,
    document_index_bytes: usize,
    retained_cst_bytes: usize,
    source_shared: bool,
    cst_shared: bool,
    lower_us: u128,
}

fn measure(elements: usize, profile: RetentionProfile) -> Measurement {
    let bytes = source(elements);
    let source_len = bytes.len();
    let parsed = ParsedFile::parse(bytes).expect("benchmark input parses");
    let started = Instant::now();
    let result = lower_proving_fixture_with_retention(
        &parsed,
        LoweringMode::Strict,
        profile,
        &SharedRegistry,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let document = result.document.expect("benchmark input lowers");
    document.validate().expect("benchmark document validates");
    let stats = document.statistics();
    assert_eq!(document.retention_profile(), profile);
    assert_eq!(
        stats.retained_source_bytes,
        match profile {
            RetentionProfile::SemanticOnly => 0,
            RetentionProfile::SyntaxOnly | RetentionProfile::Hybrid => source_len,
        }
    );
    assert_eq!(
        stats.retained_cst_nodes > 0,
        profile != RetentionProfile::SemanticOnly
    );
    assert_eq!(
        stats.retained_mapping_entries,
        match profile {
            RetentionProfile::Hybrid => 1,
            RetentionProfile::SyntaxOnly | RetentionProfile::SemanticOnly => 0,
        }
    );
    Measurement {
        source_bytes: source_len,
        semantic_attributes: stats.local_attributes,
        payload_bytes: stats.payload_blob_bytes,
        payload_blobs: stats.payload_blobs,
        cst_nodes: stats.retained_cst_nodes,
        mappings: stats.retained_mapping_entries,
        direct_owned_bytes: stats.direct_owned_bytes,
        document_index_bytes: stats.document_index_bytes,
        retained_cst_bytes: stats.retained_cst_bytes,
        source_shared: stats.source_storage_shared,
        cst_shared: stats.cst_storage_shared,
        lower_us: started.elapsed().as_micros(),
    }
}

fn main() {
    let small = [
        measure(100, RetentionProfile::SyntaxOnly),
        measure(100, RetentionProfile::SemanticOnly),
        measure(100, RetentionProfile::Hybrid),
    ];
    let large = [
        measure(10_000, RetentionProfile::SyntaxOnly),
        measure(10_000, RetentionProfile::SemanticOnly),
        measure(10_000, RetentionProfile::Hybrid),
    ];
    for (small, large) in small.iter().zip(large.iter()) {
        assert_eq!(small.semantic_attributes, large.semantic_attributes);
        assert_eq!(small.payload_blobs, large.payload_blobs);
        assert!(large.payload_bytes > small.payload_bytes * 50);
    }
    assert_eq!(small[0].cst_nodes, large[0].cst_nodes);
    assert_eq!(small[2].mappings, 1);
    assert_eq!(large[2].mappings, 1);

    println!(
        "elements,profile,source_bytes,semantic_attributes,payload_blobs,payload_bytes,cst_nodes,mappings,direct_owned_bytes,document_index_bytes,retained_cst_bytes,source_shared,cst_shared,lower_us"
    );
    for (elements, measurements) in [(100, &small), (10_000, &large)] {
        for (profile, measurement) in [
            ("SyntaxOnly", &measurements[0]),
            ("SemanticOnly", &measurements[1]),
            ("Hybrid", &measurements[2]),
        ] {
            println!(
                "{elements},{profile},{},{},{},{},{},{},{},{},{},{},{},{}",
                measurement.source_bytes,
                measurement.semantic_attributes,
                measurement.payload_blobs,
                measurement.payload_bytes,
                measurement.cst_nodes,
                measurement.mappings,
                measurement.direct_owned_bytes,
                measurement.document_index_bytes,
                measurement.retained_cst_bytes,
                measurement.source_shared,
                measurement.cst_shared,
                measurement.lower_us
            );
        }
    }
}

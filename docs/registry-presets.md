# Registry presets

Bundled presets register selected custom operation forms for structural parsing
and lowering. Each includes the core module and function grammar. Use them to
expose operands, result types, attributes, and supported regions to queries.

Load a preset in the CLI with `zirium --preset stablehlo --strict QUERY INPUT`.
Use `--list-presets` to discover names. Repeat `--preset` to combine dialects;
it can also be combined with `--registry` files. `--strict` rejects recovered
custom operations instead of warning about incomplete query information.

Load a preset in Python with `DialectRegistry.from_name("stablehlo")`, or combine
presets in a JSON registry:

```json
{
  "presets": ["scf", "arith", "memref"],
  "builtins": [],
  "operation_shapes": []
}
```

Pass the file to `zirium --registry registry.json ...` or
`DialectRegistry.from_file("registry.json")`. See [custom formats](custom-formats.md)
for configuration and API details.

## Available presets

Names below are the exact strings accepted by the configuration. Each links to
its registry definition, where you can check whether a particular operation is
registered and which grammar it uses.

| Area | Presets |
| --- | --- |
| Machine learning | [stablehlo](../crates/zirium/registries/stablehlo.json), [tosa](../crates/zirium/registries/tosa.json) |
| Functions and control flow | [func](../crates/zirium/registries/func.json), [cf](../crates/zirium/registries/cf.json), [scf](../crates/zirium/registries/scf.json), [affine](../crates/zirium/registries/affine.json), [async](../crates/zirium/registries/async.json) |
| Arithmetic | [arith](../crates/zirium/registries/arith.json), [math](../crates/zirium/registries/math.json), [complex](../crates/zirium/registries/complex.json), [index](../crates/zirium/registries/index.json), [quant](../crates/zirium/registries/quant.json) |
| Tensors and memory | [linalg](../crates/zirium/registries/linalg.json), [tensor](../crates/zirium/registries/tensor.json), [memref](../crates/zirium/registries/memref.json), [bufferization](../crates/zirium/registries/bufferization.json), [shape](../crates/zirium/registries/shape.json), [sparse_tensor](../crates/zirium/registries/sparse_tensor.json), [vector](../crates/zirium/registries/vector.json), [ptr](../crates/zirium/registries/ptr.json) |
| Parallel and distributed execution | [acc](../crates/zirium/registries/acc.json) (OpenACC), [omp](../crates/zirium/registries/omp.json) (OpenMP), [gpu](../crates/zirium/registries/gpu.json), [mpi](../crates/zirium/registries/mpi.json), [shard](../crates/zirium/registries/shard.json) |
| GPU targets | [amdgpu](../crates/zirium/registries/amdgpu.json), [nvgpu](../crates/zirium/registries/nvgpu.json), [nvvm](../crates/zirium/registries/nvvm.json), [rocdl](../crates/zirium/registries/rocdl.json), [spirv](../crates/zirium/registries/spirv.json), [xegpu](../crates/zirium/registries/xegpu.json), [xevm](../crates/zirium/registries/xevm.json) |
| CPU targets | [amx](../crates/zirium/registries/amx.json), [arm_neon](../crates/zirium/registries/arm_neon.json), [arm_sme](../crates/zirium/registries/arm_sme.json), [arm_sve](../crates/zirium/registries/arm_sve.json), [x86vector](../crates/zirium/registries/x86vector.json) |
| Lowering and program representation | [llvm](../crates/zirium/registries/llvm.json), [emitc](../crates/zirium/registries/emitc.json), [ml_program](../crates/zirium/registries/ml_program.json), [wasmssa](../crates/zirium/registries/wasmssa.json), [ub](../crates/zirium/registries/ub.json) |
| Transformation and constraints | [transform](../crates/zirium/registries/transform.json), [smt](../crates/zirium/registries/smt.json) |
| Core-only placeholders | [dlti](../crates/zirium/registries/dlti.json), [irdl](../crates/zirium/registries/irdl.json), [pdl](../crates/zirium/registries/pdl.json), [pdl_interp](../crates/zirium/registries/pdl_interp.json) |

For the installed release's names, use `DialectRegistry.preset_names()` in
Python or `DialectRegistry::preset_names()` in Rust. Core-only placeholders add
no dialect-specific custom grammar; their operations continue through recovery.
The `func` preset is also equivalent to core, which already includes its
registered function forms.

## What coverage means

An operation can have several custom spellings. Registration covers the forms
that match its grammar; optional clauses or other variants may still require
recovery. Generic quoted operations
need no preset, and dialect types and attributes generally retain their balanced
payloads as opaque values.

Some presets cover broad groups of operations: StableHLO includes elementwise
operations, typed signatures, and region-bearing forms such as `reduce`, `sort`,
and `while`; SCF includes structured regions and loop header bindings; Linalg
includes structured and generated named operations. Other presets cover a small
subset. For example, `wasmssa` registers only `wasmssa.return` beyond core.

Common limits affect how you can use these presets:

- Compact signatures that infer operand or result types may require recovery.
  A printed container or operand type cannot safely be treated as a result type.
- Positional modifiers, mixed static/dynamic indices, and dictionaries after
  regions or type trailers may fall outside the registered grammar. For example,
  several Arith operations accept their default form but recover variants with
  positional overflow, exact, or fast-math modifiers.
- Unsupported custom forms use best-effort recovery. Inspect parsing and
  lowering diagnostics before relying on semantic structure or editing output.
- Compact StableHLO reductions using `applies` retain their explicit textual
  structure without synthesizing an implicit reducer body. Dot dimension pairs
  retain both sides in attribute projection and print generically as nested
  arrays. These are structural representations of the custom clauses.

Presets provide structural support. Full dialect verification, type inference,
execution, target checks, and serialization are outside their scope. StableHLO
support does not include VHLO or portable-artifact compatibility.

The preset definitions were checked against LLVM 22.1.0, except StableHLO,
which was checked against StableHLO 1.20.1. Preset names are unversioned and track
Zirium releases; they do not claim complete coverage of those upstream versions.

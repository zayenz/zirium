# Custom formats across Rust, Python, and the CLI

Zirium separates syntax recovery from semantic support. Recovering an unknown
custom operation lets a tool inspect its name, source text, and nested regions.
It does not establish that the operation can be verified or rewritten.

## Registry choices

| Registry | Custom syntax |
| --- | --- |
| Empty, including the default Python parser | Unknown operations are recovered; generic quoted operations are parsed normally. |
| Core | `builtin.module`, `func.func`, `func.call`, and `func.return`. |
| Proving | Core plus `arith.constant`, `arith.addi`, `cf.br`, and `cf.cond_br`. |
| StableHLO preset | Core plus 96 common StableHLO custom forms. |
| TOSA preset | Core plus 93 TOSA tensor, shape, control-flow, and utility forms. |
| SCF preset | Core plus 11 of 12 SCF operations, including structured regions and loop header bindings. |
| Linalg preset | Core plus all 99 core, structured, relayout, and generated named Linalg operations. |
| OpenACC preset | Core plus 35 mapping, bounds-accessor, region, and terminator forms. |
| Affine preset | Core plus 4 of the 16 Affine operations. |
| AMDGPU preset | Core plus 4 of the 33 AMDGPU operations. |
| AMX preset | Core plus 1 of the 5 AMX operations. |
| Arith preset | Core plus default forms of 43 of the 51 Arith operations. |
| ArmNeon preset | Core plus 1 of the 7 ArmNeon operations. |
| ArmSME preset | Core plus 3 of the 68 ArmSME operations. |
| ArmSVE preset | Core plus 9 of the 21 ArmSVE custom-form operations. |
| Async preset | Core plus 11 of the 29 Async operations. |
| Bufferization preset | Core plus 5 of the 7 Bufferization operations. |
| CF preset | Core plus `cf.br` and `cf.cond_br`. |
| Complex preset | Core plus default forms of 21 of the 29 Complex operations. |
| DLTI preset | Core only; all 6 DLTI attributes remain opaque dialect values. |
| EmitC preset | Core plus 20 of the 45 EmitC custom-form operations. |
| Func preset | Exact custom forms for 3 of the 5 Func operations. |
| GPU preset | Core plus default forms of 10 of the 66 GPU operations. |
| Index preset | Core plus the 2 explicitly typed casts among 26 Index operations. |
| IRDL preset | Core only; all 17 IRDL operations remain on recovery. |
| LLVM preset | Core plus 138 explicit core and intrinsic forms among 284 LLVM operations. |
| MemRef preset | Core plus 11 structurally exact forms among 32 MemRef operations. |
| MLProgram preset | Core only; all 11 MLProgram operations remain on recovery. |
| MPI preset | Core plus 4 structurally exact forms among 15 MPI operations. |
| NVGPU preset | Core plus 7 structurally exact forms among 24 NVGPU operations. |
| NVVM preset | Core plus 71 structurally exact forms among 185 NVVM operations. |
| OpenMP preset | Core plus 8 structurally exact forms among 54 OpenMP operations. |
| PDL preset | Core only; all 15 PDL operations remain on recovery. |
| PDLInterp preset | Core only; all 39 PDLInterp operations remain on recovery. |
| Ptr preset | Core plus default forms of 4 among 13 Ptr operations. |
| ROCDL preset | Core plus 125 structurally exact forms among 323 ROCDL operations. |
| Shard preset | Core plus 1 structurally exact form among 22 Shard operations. |
| Declarative | A selected subset of the proving catalog. |
| Operation shapes | Caller-named operations using one of the supported structural grammars. |

Core, proving, and declarative registries containing `builtin.module` accept
`module` as its shorthand, including named and nested modules. The empty
registry does not enable this grammar.

The declarative registry selects existing implementations. It does not
interpret arbitrary MLIR assembly-format strings or load ODS/TableGen files.
Rust callers can also construct static descriptors with parser, lowering,
verification, and printing callbacks. Python does not expose those callbacks.

The bundled `stablehlo` preset provides structural lowering for 96 custom
forms: unary and binary elementwise operations, constants, typed returns,
ordinary variadic signatures, and operations that mix operands with fixed or
named clauses before a trailing type signature. The latter group includes
`broadcast_in_dim`, `compare`, `concatenate`, `convolution`, `custom_call`,
`dot_general`, dynamic slices, `iota`, `pad`, `select`, `slice`, and
`transpose`. Their operands, result types, ordinary attribute dictionaries,
and simple `name = value` clauses are available to semantic and CLI queries.

The preset also retains the regions and block arguments of 14 region-bearing
forms, including `reduce`, `reduce_window`, `scatter`, `sort`, and `while`.
This is structural support, not a StableHLO implementation: Zirium does not
apply StableHLO verification, type inference, execution semantics, VHLO, or
portable-artifact compatibility. Unsupported custom forms continue through
best-effort recovery, and generic quoted StableHLO operations need no preset. The embedded
[registry file](../crates/zirium/registries/stablehlo.json) is the exact preset
definition. This surface was checked against StableHLO 1.20.1. The unversioned
preset name tracks Zirium releases; it does not claim support for the complete
StableHLO 1.20.1 opset or its portable artifact format.

The `tosa`, `scf`, `linalg`, `acc`, `affine`, `amdgpu`, `amx`, `arith`,
`arm_neon`, `arm_sme`, `arm_sve`, `async`, `bufferization`, `cf`, `complex`, `dlti`, `emitc`, `func`, `gpu`, `index`, `irdl`, `llvm`, `math`, `memref`, `ml_program`, `mpi`, `nvgpu`, `nvvm`, `omp`, `pdl`, `pdl_interp`, `ptr`, `quant`, `rocdl`, and `shard` presets were checked against
LLVM 22.1.0.
TOSA registers 93 of the 94 operations defined by its main, utility, and shape
operation files; `tosa.variable` remains on the generic recovery path because
its custom symbol/type form has no reusable structural signature. SCF registers
all 12 operations. Linalg registers all 99 operations: its 16 core/structured
operations, both relayout operations, and 81 generated named operations.
Tensor-result forms expose their trailing result types; buffer forms without a
result signature remain usable through recovery where their custom spelling has
no safe structural boundary.

OpenACC registers 35 of its 54 operations. This includes all 16 data-entry
mapping operations, the four bounds accessors, 12 single-region constructs,
the single-region form of `acc.private.recipe`, and both terminators. Mapping
forms with a trailing `attributes` dictionary, loop result forms, and region
forms with trailing attributes use whole-operation recovery. The other 19
operations remain on that recovery path: `acc.bounds`, `acc.atomic.read`,
`acc.atomic.write`, `acc.copyout`, `acc.delete`, `acc.detach`,
`acc.update_host`, `acc.firstprivate.recipe`, `acc.reduction.recipe`,
`acc.enter_data`, `acc.exit_data`, `acc.declare_enter`, `acc.declare_exit`,
`acc.routine`, `acc.init`, `acc.shutdown`, `acc.set`, `acc.update`, and
`acc.wait`. Their custom spellings either have no safe trailing boundary in
the current shapes, require keyword-separated regions, or would incorrectly
imply SSA result types.

Affine registers 4 of its 16 operations: `affine.for`, `affine.if`,
`affine.linearize_index`, and `affine.yield`. The two structured control-flow
forms expose their regions, header bindings, operands, and result slots;
trailing attribute dictionaries after the final region use whole-operation
recovery. The linearization form exposes its index operands and single index
result, while the terminator preserves its optional typed operands. The other
12 operations remain on the recovery path: `affine.apply`, `affine.min`,
`affine.max`, `affine.parallel`, `affine.load`, `affine.store`,
`affine.vector_load`, `affine.vector_store`, `affine.prefetch`,
`affine.delinearize_index`, `affine.dma_start`, and `affine.dma_wait`.
Their custom forms either lack a safe trailing boundary, need parallel header
bindings, derive a result type from a memref element type, or spell types for a
no-result operation. Assigning the current clause shapes to those forms would
lose structure or invent semantic result types.

AMDGPU registers 4 of its 33 concrete operations: `amdgpu.ext_packed_fp8`,
`amdgpu.scaled_ext_packed_matrix`,
`amdgpu.tensor_load_to_lds`, and `amdgpu.tensor_store_from_lds`. These expose
their SSA operands and real result slots without treating accumulator,
container, or index types as results. The three `!amdgpu.tdm_*` types and the
`#amdgpu.address_space`, `#amdgpu.dpp_perm`,
`#amdgpu.sched_barrier_opt`, and `#amdgpu.mfma_perm_b` attributes remain opaque
dialect values, as they do without the preset.

The other 29 operations remain on whole-operation recovery:
`amdgpu.scaled_ext_packed`, `amdgpu.packed_trunc_2xfp8`,
`amdgpu.packed_scaled_trunc`, `amdgpu.packed_stoch_round_fp8`,
`amdgpu.fat_raw_buffer_cast`, `amdgpu.raw_buffer_load`,
`amdgpu.raw_buffer_store`, the five `amdgpu.raw_buffer_atomic_*` operations,
`amdgpu.dpp`, `amdgpu.swizzle_bitmode`, `amdgpu.permlane_swap`,
`amdgpu.lds_barrier`, `amdgpu.sched_barrier`,
`amdgpu.memory_counter_wait`, `amdgpu.mfma`, `amdgpu.wmma`,
`amdgpu.sparse_mfma`, `amdgpu.gather_to_lds`, `amdgpu.transpose_load`,
`amdgpu.scaled_mfma`, `amdgpu.scaled_wmma`,
`amdgpu.make_gather_dma_base`, `amdgpu.make_dma_base`,
`amdgpu.make_gather_dma_descriptor`, and `amdgpu.make_dma_descriptor`.
Their compact type trailers omit inferred operand types or name container and
accumulator types rather than SSA results, or their required positional
attributes and attribute-only forms lack a matching reusable shape.
`amdgpu.sched_barrier` and `amdgpu.memory_counter_wait` also lack a typed
trailing boundary. Registering these operations with the current broad clause
shape would therefore assign incorrect semantic types or results. AMDGPU
defines no regions or successors; none of its operations need post-region
dictionaries or tuple/custom header bindings. The dependent `rocdl` namespace
is a separate LLVM IR dialect and is not part of the 33-operation AMDGPU count.

AMX registers 1 of its 5 operations: `amx.tile_zero`. It exposes the operation's
single tile result, including the opaque `!amx.tile<...>` spelling. The other
four operations remain on whole-operation recovery: `amx.tile_load`,
`amx.tile_store`, `amx.tile_mulf`, and `amx.tile_muli`. Load uses an `into`
type trailer; store's trailer names operand types despite having no result; and
the multiply trailers name all three operand types while the result type is
inferred from the accumulator. Treating those trailers as ordinary result
signatures would invent results or assign incorrect types. The optional `zext`
keywords on `amx.tile_muli` add a second unsupported header detail. AMX defines
the single `!amx.tile` type and no attributes; the type remains an opaque
dialect value, as it does without the preset. AMX operations have no regions or
successors, so post-region dictionaries do not apply.

Arith registers the default forms of 43 of its 51 operations. This includes
the built-in proving implementations of `arith.constant` and `arith.addi`, 28
ordinary binary operations, and 13 unary or cast operations. The reusable
unary and binary shapes expose exact operand counts and real result types,
including the source and destination types of casts. The preset does not add
Arith type registrations because the dialect defines no types. Its
`#arith.fastmath` and `#arith.overflow` attributes remain opaque dialect values;
the built-in `arith.addi` implementation additionally exposes its positional
overflow flags as the `overflowFlags` attribute.

Eight operations remain on whole-operation recovery: `arith.addui_extended`,
`arith.mulsi_extended`, `arith.mului_extended`, `arith.scaling_extf`,
`arith.scaling_truncf`, `arith.cmpi`, `arith.cmpf`, and `arith.select`. The
three extended integer operations have two results whose relationship to the
compact type trailer is not representable by the current unary and binary
shapes. The scaling casts spell two independently typed operands before a
single result type. Comparisons place a predicate before their operands and
infer an `i1`-shaped result from the operand type. Select has three operands,
and its one- or two-type trailer does not always spell the condition type.
Using the broad clause shape for these operations would therefore assign
incorrect semantic types or omit a required positional attribute.

The registered default forms of `arith.subi`, `arith.muli`, `arith.shli`, and
`arith.trunci` do not include positional `overflow` clauses. The default forms
of `arith.divui`, `arith.divsi`, `arith.shrui`, and `arith.shrsi` do not include
`exact`; and the floating-point operations do not include positional
`fastmath` or rounding-mode modifiers. Those variants continue through
whole-operation recovery. `arith.constant` supports the proving implementation's
typed scalar spelling; inferred boolean constants, shaped constants, and a
dictionary printed before the value may require recovery or fail strict Arith
verification. Arith operations have no regions or successors, so post-region
dictionaries do not apply.

ArmNeon registers 1 of its 7 operations: `arm_neon.intr.smull`. Its two
operands share the source-vector type in the trailer, and the type after `to`
is its single widened-vector result, so the binary shape exposes both operands
and the real result without inference. The ordinary attribute dictionary is
also available to semantic queries.

The other six operations remain on whole-operation recovery:
`arm_neon.intr.sdot`, `arm_neon.intr.smmla`, `arm_neon.intr.ummla`,
`arm_neon.intr.usmmla`, `arm_neon.intr.bfmmla`, and `arm_neon.2d.sdot`.
Each has three operands. The `sdot` forms spell the two independently typed
dot-product operands and infer the accumulator type from the result; the four
matrix-multiply forms spell only their shared source type and infer the
accumulator type from the result. The current variadic shape would therefore
assign incorrect types to operands. ArmNeon defines no dialect types or
attributes, and its operations have no regions or successors, so trailing
region dictionaries do not apply.

ArmSME registers 3 of its 68 concrete operations: `arm_sme.get_tile`,
`arm_sme.zero`, and `arm_sme.copy_tile`. The first two expose their single
tile result, and copy exposes its one same-typed tile operand and result.
Ordinary attribute dictionaries are preserved. ArmSME defines no dialect
types; its scalable vector tiles use the builtin vector type. Its three
`#arm_sme.layout`, `#arm_sme.kind`, and `#arm_sme.type_size` enum attributes
remain opaque dialect attributes, as they do without the preset.

The other 22 high-level operations remain on whole-operation recovery. Tile
loads and stores and the four slice operations interleave indexed operands,
optional masks, and optional positional `layout` modifiers. Insert and extract
use bracketed slice indices, optional `layout`, and `into` or `from` type
trailers. `arm_sme.outerproduct` omits its derived result type and may add
positional `kind`, accumulator, and mask clauses. The 14 widening outer-product
operations have a structurally useful `into` trailer in their default form,
but their optional accumulator type is inferred from the result and mask types
are omitted. Registering their full forms with an existing shape would assign
incorrect operand types. `arm_sme.streaming_vl` has a positional type-size
attribute and an inferred `index` result, so it has no typed trailing boundary.

The remaining 43 operations are LLVM intrinsic wrappers with generic quoted
syntax and therefore need no custom-form registration. They comprise intrinsic
zero, 16 MOP variants, 10 loads, 10 stores, `str`, two writes, two reads, and
`cntsd`. None of ArmSME's operations have regions or successors, so
post-region dictionaries do not apply. Declarative attribute dictionaries sit
before typed trailers; `streaming_vl` instead permits a trailing dictionary,
but still has no typed boundary from which to recover its inferred result.

ArmSVE registers all nine operations instantiated from its scalable masked
integer and floating-point bases: `arm_sve.masked.addi`,
`arm_sve.masked.addf`, `arm_sve.masked.subi`, `arm_sve.masked.subf`,
`arm_sve.masked.muli`, `arm_sve.masked.mulf`,
`arm_sve.masked.divi_signed`, `arm_sve.masked.divi_unsigned`, and
`arm_sve.masked.divf`. Each exposes exactly three operands and one result. The
two trailer types map to the mask and the shared source/result vector type, so
the semantic function type retains the real `(mask, source, source) -> result`
relationship. Ordinary attribute dictionaries are preserved, including on
multi-dimensional scalable-vector forms.

The other 12 custom-form operations remain on whole-operation recovery. The
five integer dot/matrix forms and `arm_sve.intr.bfmmla` infer the accumulator
type from the result type. `arm_sve.convert_from_svbool` and
`arm_sve.convert_to_svbool` spell only the type from which the other side is
derived. `arm_sve.zip.x2` and `arm_sve.zip.x4` return two and four same-typed
results, respectively. `arm_sve.psel` interleaves an indexed operand and spells
two independently mapped predicate types, while `arm_sve.dupq_lane` has a
positional lane attribute in brackets. The current reusable shapes cannot
represent those relationships without losing structure or assigning an
incorrect type.

ArmSVE also defines 21 LLVM intrinsic wrappers whose canonical syntax is the
generic quoted operation form and needs no custom registration. This count is
separate from `arm_sve.intr.bfmmla`, which overrides the intrinsic base with a
custom assembly format and is one of the 12 gaps above. ArmSVE defines no
dialect types or attributes, and none of its operations have regions or
successors. Transform-dialect pattern operations with `transform.*` names are
outside the `arm_sve` operation namespace and this preset.

Async registers 11 of its 29 operations. `async.func` and `async.call` expose
their symbols, signatures, operands, wrapped result types, and function body;
the structural shapes do not add Async symbol-use verification. `async.return`
and `async.yield` expose their unwrapped, typed operands and no SSA results.
The return form with both a leading attribute dictionary and operands remains
on whole-operation recovery because that dictionary precedes the operand list;
the ordinary yield dictionary follows it and is supported.

The preset also registers `async.runtime.create`,
`async.runtime.num_worker_threads`, `async.runtime.set_available`,
`async.runtime.set_error`, `async.runtime.await`, `async.runtime.add_ref`, and
`async.runtime.drop_ref`. These forms expose only types that their syntax
actually spells: real result types for the two result-only operations, and
operand types with no invented results for the other five. Reference-count
attributes in the ordinary dictionary are retained.

The other 18 operations remain on whole-operation recovery: `async.execute`,
`async.await`, `async.create_group`, `async.add_to_group`, `async.await_all`,
the six `async.coro.*` operations, `async.runtime.create_group`,
`async.runtime.is_error`, `async.runtime.resume`,
`async.runtime.await_and_resume`, `async.runtime.store`,
`async.runtime.load`, and `async.runtime.add_to_group`. Their custom forms use
execute-specific dependency and body bindings, infer token/value/group/coroutine/
index/i1 results, omit some operand types, or (for `async.coro.suspend`) carry
three successors. A nearby reusable shape would lose those roles or invent
types. Async defines no dialect attributes. Its six `!async.*` types remain
opaque, including the wrapped element type of `!async.value<...>`. The only
operation regions are the supported function body and unsupported execute
body; neither operation uses a post-region dictionary.

Bufferization registers 5 of its 7 operations. `bufferization.clone` exposes
its memref operand and its independently spelled memref result type.
`bufferization.dealloc_tensor` exposes one typed tensor operand and no SSA
result. `bufferization.to_tensor` and `bufferization.to_buffer` expose their
single operand and real conversion result; their optional `restrict`,
`writable`, and `read_only` unit-keyword clauses remain source-preserved rather
than becoming semantic dictionary attributes. `bufferization.materialize_in_destination`
exposes its source and destination operands and uses its full function-type
trailer, so tensor destinations have the explicitly spelled tensor result and
memref destinations correctly have no result. Ordinary attribute dictionaries
on these forms remain available to semantic queries.

The other two operations remain on whole-operation recovery.
`bufferization.alloc_tensor` mixes index-valued dynamic sizes, an optional
tensor copy, and an optional index size hint, but its trailer spells only the
inferred tensor result. `bufferization.dealloc` types its memref lists while
omitting the `i1` condition types and inferring one `i1` result per retained
memref. Registering either with the broad clause shape would assign incorrect
operand or result types. All 7 operations use custom assembly; there are no
additional default/generic concrete operations. The dialect defines no types
or dialect attributes, and none of its operations have regions or successors.

CF registers 2 of its 4 operations. The dedicated `cf.br` and `cf.cond_br`
implementations retain successor targets, typed successor arguments, ordinary
attribute dictionaries, and block resolution. All four CF operations have no
results or regions. The `branch_weights` attribute on `cf.cond_br` is retained
and checked by the built-in semantic verifier when written in the trailing
dictionary; the positional `weights(...)` variant remains on recovery.

`cf.assert` and `cf.switch` remain on whole-operation recovery. Assert has a
required positional string attribute but no type trailer, which is outside the
current reusable formats. Switch combines a typed flag with a bracketed custom
case list, variadic successors, successor operands, and case attributes.
Approximating either operation with a broad shape would lose required semantic
structure. CF defines no dialect types or attributes.

Complex registers the default forms of 21 of its 29 operations. Thirteen unary
operations whose operand and result have the same `complex<T>` type use the
unary shape, and six arithmetic operations whose two operands and result share
that type use the binary shape. `complex.bitcast` uses the unary shape's full
source-`to`-destination trailer. `complex.constant` uses the literal shape to
retain its positional array value and explicitly spelled result type. These
forms expose their operands, real result types, values, and ordinary attribute
dictionaries. Their optional positional `fastmath` modifiers are not part of
the registered default spelling and continue through whole-operation recovery.

Eight operations remain on whole-operation recovery. `complex.abs`,
`complex.im`, `complex.re`, and `complex.angle` derive a floating-point result
from the element type of their complex operand. `complex.create` spells its
complex result type but infers the two floating-point operand types;
`complex.eq` and `complex.neq` spell their operand type but infer an `i1`
result; and `complex.powi` has independently typed complex and integer operands.
Registering these with a nearby shape would assign incorrect types.

The `complex<T>` type is a builtin MLIR type, not a Complex dialect type. The
dialect's `#complex.number` attribute remains an opaque dialect value, as it
does without the preset. None of the 29 operations have regions or successors;
their declarative forms place ordinary dictionaries before the type trailer.

DLTI defines no operations or types in LLVM 22.1. It defines six attributes:
`#dlti.dl_entry`, `#dlti.dl_spec`, `#dlti.map`,
`#dlti.target_system_spec`, `#dlti.target_device_spec`, and
`#dlti.function_pointer_alignment`. Zirium already parses and lowers their
balanced spellings as opaque dialect attributes, so the `dlti` preset adds core
syntax without registering operations or claiming attribute verification.
`transform.dlti.query` belongs to the Transform dialect and is outside this
preset.

EmitC defines 49 concrete operations in LLVM 22.1. Four of them use only the
generic quoted operation form: `emitc.constant`, `emitc.variable`,
`emitc.member`, and `emitc.member_of_ptr`. They need no custom registration.
The preset registers 20 of the remaining 45 custom forms. Thirteen ordinary
unary or binary operators (`emitc.add`, the six `emitc.bitwise_*` operations,
`emitc.div`, `emitc.mul`, `emitc.rem`, `emitc.sub`, `emitc.unary_minus`, and
`emitc.unary_plus`) expose their complete functional type. `emitc.cast` also
exposes its source and destination types. `emitc.literal` retains its string
value and result type, while `emitc.call` retains its symbol callee, operands,
and complete function type. `emitc.subscript` retains the container and index
operands plus its complete function type. `emitc.return` and `emitc.yield`
expose their optional typed operand without inventing a result.

The ordinary public spelling of `emitc.func` is registered as a function-like
form, including its signature, argument and result dictionaries, operation
attributes, and optional body. Visibility-prefixed variants such as
`emitc.func private` remain on whole-operation recovery. A leading attribute
dictionary before a typed `emitc.return` or `emitc.yield` operand also remains
on recovery; the attribute-free forms and dictionary-only zero-operand forms
are covered.

The other 25 custom forms remain on whole-operation recovery: `emitc.file`,
`emitc.address_of`, `emitc.apply`, `emitc.call_opaque`, `emitc.cmp`,
`emitc.dereference`, `emitc.expression`, `emitc.for`, `emitc.declare_func`,
`emitc.include`, the three `emitc.logical_*` operations, `emitc.load`,
`emitc.conditional`, `emitc.global`, `emitc.get_global`, `emitc.verbatim`,
`emitc.assign`, `emitc.if`, `emitc.switch`, `emitc.class`, `emitc.field`,
`emitc.get_field`, and `emitc.do`. Their syntax either infers an operand or
result type, carries a required positional attribute or symbol outside the
current semantic captures, or uses operation-specific region structure.
Registering a nearby broad shape would lose a required role or assign an
incorrect type. EmitC defines seven dialect types and two attributes; their
balanced `!emitc.*` and `#emitc.*` spellings already remain lossless opaque
values. None of the operations has successors. The only registered region is
the ordinary `emitc.func` body; all operation-specific region forms recover.

Func defines five operations in LLVM 22.1. The preset registers the existing
exact implementations of `func.func`, `func.call`, and `func.return`. They
preserve function symbols and signatures, direct-call callees and functional
types, typed return operands, function bodies, and ordinary attribute
dictionaries.

`func.constant` and `func.call_indirect` remain on whole-operation recovery.
The constant has a leading attribute dictionary followed by a symbol value and
a result type. The indirect call has an SSA callee followed by argument
operands, while its result types are derived from the callee's function type.
Current broad operation shapes would lose those distinct roles or claim the
wrong result structure. Func defines no dialect types or attributes, and none
of its operations has successors.

GPU defines 66 concrete operations in LLVM 22.1, all with custom assembly;
there are no generic-only operation definitions to count separately. The
preset registers 10 exact default forms. `gpu.subgroup_id`,
`gpu.num_subgroups`, `gpu.subgroup_size`, and
`gpu.dynamic_shared_memory` expose their explicitly spelled result type and no
operands. The optional `upper_bound` variants of the first three remain on
recovery. `gpu.return` and `gpu.yield` expose typed operands and no results;
`gpu.terminator` and `gpu.barrier` expose their dictionary-only zero-operand,
zero-result forms. `gpu.host_register` and `gpu.host_unregister` expose their
single typed memref operand without inventing an SSA result.

The other 56 operations remain on whole-operation recovery. Ten index-query
forms (`gpu.cluster_dim`, `gpu.cluster_dim_blocks`, `gpu.cluster_id`,
`gpu.cluster_block_id`, `gpu.block_dim`, `gpu.block_id`, `gpu.grid_dim`,
`gpu.thread_id`, `gpu.global_id`, and `gpu.lane_id`) infer their index result
instead of spelling its type. Six symbol, launch, and region forms (`gpu.func`,
`gpu.launch_func`, `gpu.launch`, `gpu.module`, `gpu.binary`, and
`gpu.warp_execute_on_lane_0`) need GPU-specific symbols, async dependency
segments, grid/block/cluster clauses, workgroup/private attributions, or region
header bindings. Their roles do not match the conventional function, call, or
module shapes.

Seven miscellaneous forms (`gpu.printf`, `gpu.all_reduce`,
`gpu.subgroup_reduce`, `gpu.shuffle`, `gpu.rotate`, `gpu.set_default_device`,
and `gpu.subgroup_broadcast`) use literal strings, positional enum or integer
attributes, inferred result types, indexed clauses, or optional reduction
regions. The five async memory forms (`gpu.wait`, `gpu.alloc`, `gpu.dealloc`,
`gpu.memcpy`, and `gpu.memset`) couple bracketed dependency lists to optional
token results; alloc additionally separates dynamic and symbol operands. The
seven subgroup-MMA forms (`gpu.subgroup_mma_load_matrix`,
`gpu.subgroup_mma_store_matrix`, `gpu.subgroup_mma_compute`,
`gpu.subgroup_mma_constant_matrix`, `gpu.subgroup_mma_extract_thread_local`,
`gpu.subgroup_mma_insert_thread_local`, and
`gpu.subgroup_mma_elementwise`) use indexed operands, positional operation
attributes, or compact trailers that infer matrix operand/result types.

The remaining 21 sparse-runtime forms are `gpu.create_dn_tensor`,
`gpu.destroy_dn_tensor`, `gpu.create_coo`, `gpu.create_coo_aos`,
`gpu.create_csr`, `gpu.create_csc`, `gpu.create_bsr`,
`gpu.create_2to4_spmat`, `gpu.destroy_sp_mat`, `gpu.spmv_buffer_size`,
`gpu.spmv`, `gpu.spmm_buffer_size`, `gpu.spmm`, `gpu.sddmm_buffer_size`,
`gpu.sddmm`, `gpu.spgemm_create_descr`, `gpu.spgemm_destroy_descr`,
`gpu.spgemm_work_estimation_or_compute`, `gpu.spgemm_copy`,
`gpu.spmat_get_size`, and `gpu.set_csr_pointers`. These forms combine async
tokens with sparse handles, compute types and transpose/action enums, and in
some cases multiple inferred index or token results. Together with the
preceding groups, this is the complete 56-operation gap.

GPU's five dialect types (`!gpu.async.token`, `!gpu.mma_matrix`, and the three
sparse handle types) and its dialect attributes remain balanced opaque values.
The ten operations in `GPUTransformOps.td` have `transform.*` names, including
the `transform.gpu.*` mapping operations and the GPU conversion/rewrite pattern
descriptors; they belong to the Transform dialect and are outside the 66-op
GPU namespace inventory and this preset.

Index defines 26 operations in LLVM 22.1. The preset registers `index.casts`
and `index.castu`: each exposes one operand, its source type, its destination
result type, and an ordinary attribute dictionary. The source-to-destination
syntax matches the reusable unary shape exactly. Signed versus unsigned
extension is dialect semantics rather than a difference in structural shape.

The other 24 operations remain on whole-operation recovery. All 20 arithmetic
and bitwise binary operations (`index.add`, `index.sub`, `index.mul`, the five
division forms, the two remainder forms, the four min/max forms, the three
shift forms, and `index.and`, `index.or`, and `index.xor`) omit their operand
and inferred index result types. `index.cmp` has a positional predicate and an
inferred `i1` result. `index.sizeof` infers its index result, while
`index.constant` and `index.bool.constant` combine inferred result types with
positional attributes. A nearby typed shape would invent semantics for these
forms, so the preset does not register them.

Index defines no dialect types; `index` is a builtin MLIR type. Its one dialect
attribute is the comparison-predicate enum used positionally by `index.cmp`.
All 26 operations admit ordinary attribute dictionaries, and none has regions
or successors. Dictionaries on unsupported operations remain lossless inside
whole-operation recovery; the preset does not interpret the predicate or
constant attributes.

IRDL defines 17 operations in LLVM 22.1. The preset adds core syntax but no
operation registrations. Its four symbol-bearing definitions (`irdl.dialect`,
`irdl.type`, `irdl.attribute`, and `irdl.operation`) combine a symbol, an
optional `attributes` dictionary, and a custom single-block region. Mapping
them to the superficially similar function or region shapes would misstate
their header and region structure.

The five definition-list operations (`irdl.parameters`, `irdl.operands`,
`irdl.results`, `irdl.attributes`, and `irdl.regions`) use custom named-value
lists or a custom string-to-value map; the operand and result lists also carry
per-entry `single`, `optional`, or `variadic` markers. `irdl.region` has four
printed variants for optional entry-block constraints and block count, while
its `!irdl.region` result is inferred rather than printed. The seven constraint
operations (`irdl.is`, `irdl.base`, `irdl.parametric`, `irdl.any`,
`irdl.any_of`, `irdl.all_of`, and `irdl.c_pred`) likewise infer their
`!irdl.attribute` results; several additionally use positional attributes,
symbol references, or untyped operand lists. Registering any of these with a
nearby typed shape would invent result types or discard meaningful clauses, so
all 17 custom forms remain on whole-operation recovery.

IRDL separately defines two types, `!irdl.attribute` and `!irdl.region`, and
two attributes, the `irdl.variadicity` enum and its
`irdl.variadicity_array` container. Their namespaced spellings remain balanced
opaque dialect values, including the generic `#irdl<...>` attribute spelling;
the preset does not interpret an IRDL definition as an ODS schema and does not
apply declarations to later operations.

LLVM defines 80 concrete core operations in `LLVMOps.td` and 204 concrete
intrinsic operations in `LLVMIntrinsicOps.td` in LLVM 22.1. The count excludes
the two helper type records next to the core operations. The preset registers
43 core and 95 intrinsic forms. This inventory was checked from the concrete
TableGen definitions, their inherited generated assembly families, the
handwritten parsers and printers in `LLVMDialect.cpp`, the LLVM dialect docs,
and representative `mlir/test/Dialect/LLVMIR` files.

The core coverage includes the default forms of all 18 integer and floating
binary arithmetic operations, `llvm.fneg`, all 13 casts, `llvm.alloca`,
`llvm.va_arg`, `llvm.select`, `llvm.freeze`, three terminators, and the four
typed `llvm.mlir.none`, `undef`, `poison`, and `zero` producers. Optional
overflow, exact, disjoint, non-negative, and dereferenceability clauses are not
claimed: those variants recover as whole operations.

The intrinsic coverage follows assembly families rather than duplicating an
intrinsic-by-intrinsic grammar. It includes the parenthesized full-function-type
unary, binary, ternary, rounding, comparison, two-result, and saturation math
families; selected pointer lifetime and invariant operations; the four
constrained conversion forms; `ssa.copy` and `expect`; explicit coroutine,
variadic-call, exception, and stack forms; four floating vector reductions;
matrix multiply; masked load and gather; traps; and `stepvector`. Every
registered form spells enough operand and result type information for the
existing structural lowering. Bare `!llvm.*` types and builtin fixed or
scalable vectors use the ordinary type parser and remain queryable as typed or
opaque values as appropriate.

The other 146 operations remain on whole-operation recovery or use quoted
generic syntax. The grouped gaps are globals, aliases, functions, comdats,
symbols, and metadata; GEP and other indexed or position-derived forms; calls,
invokes, switches, indirect branches, and successor-bearing terminators;
loads, stores, fences, atomics, orderings, and operand bundles; debug metadata;
and intrinsic families whose result types are inferred or whose generated form
is generic-only, including the VP family. `llvm.intr.matrix.transpose` has an
`into` conversion, but `llvm.intr.get.active.lane.mask`, vector insertion and
extraction, and masked stores also need positional or multiple typed clauses.
Registering a nearby six-step conversion or broad clause shape for those forms
would assign the wrong structural meaning, so the preset stays conservative.
LLVM attributes, metadata, and types remain balanced opaque dialect values;
the preset does not apply LLVM verification, data-layout rules, or execution
semantics.

Math defines 46 concrete operations in LLVM 22.1, all with generated custom
assembly. The preset registers 40 structurally exact forms: 34 integer or
floating unary operations, four integer or floating binary operations, and the
two floating ternary operations `math.clampf` and `math.fma`. The unary and
binary forms use narrow shapes because their default spelling gives every
operand and the single result the same trailing type. The ternary forms use
`operand_clauses`; their single trailing type likewise maps exactly to all three
operands and the one result, including `clampf`'s `to [min, max]` punctuation.
Ordinary trailing attribute dictionaries remain queryable. Positional
`fastmath<...>` variants of the narrow unary and binary forms recover as whole
operations. The ternary clause shape accepts that positional spelling while
retaining its operands and type signature, but does not interpret the modifier
as a semantic attribute.

Six operations remain on whole-operation recovery. `math.isfinite`,
`math.isinf`, `math.isnan`, and `math.isnormal` print only their floating
operand type; their scalar `i1` or shaped boolean result is derived from the
operand shape. `math.sincos` similarly prints one operand type while inferring
two same-typed results. `math.fpowi` prints distinct floating base and integer
power types while inferring one result matching the base. The broad clause
lowering would turn its two printed types into two result slots, so registering
it would be structurally false. Math defines no dialect types or attributes;
its fast-math attribute belongs to Arith and otherwise remains an opaque
balanced value. None of the 46 operations has regions or successors, and the
preset adds no Math-specific verification, inference, or execution semantics.

MemRef defines 32 operations in LLVM 22.1. This pinned inventory differs from
the earlier expected count of 31. The preset registers 11 forms whose operand,
result, and type roles fit existing reusable shapes: `memref.assume_alignment`,
`memref.distinct_objects`, `memref.alloca_scope`,
`memref.alloca_scope.return`, `memref.cast`, `memref.dealloc`,
`memref.extract_aligned_pointer_as_index`, `memref.generic_atomic_rmw`,
`memref.atomic_yield`, `memref.memory_space_cast`, and `memref.reshape`.
The two casts are exact unary conversions. The scope and generic atomic forms
retain their regions and result slots while leaving locally inferred types
opaque. Typed terminators and `dealloc` expose operands without inventing SSA
results. `assume_alignment` has one same-typed result in the pinned definition.

The other 21 operations remain on whole-operation recovery. Allocation and
view forms mix memrefs with dynamic index and symbol operands; load, dimension,
rank, and metadata forms infer results; store, copy, DMA, wait, and prefetch
spell operand types despite having no results. Reinterpret, subview, and expand
shape use mixed static/dynamic index lists. Collapse shape needs reassociation
syntax, transpose contains an affine-map arrow, and globals need symbol-specific
headers. Registering these with the broad clause lowering would assign memref
types to index operands or turn operand-only trailers into result slots. MemRef
defines no dialect types or attributes: `memref<...>` is a builtin type, so no
value descriptors are added. The preset does not apply MemRef verification,
aliasing, layout, memory-space, or execution semantics.

MPI defines 15 operations in LLVM 22.1. The preset registers four forms whose
complete operand and result signatures fit existing shapes. Result-bearing
`mpi.init` and `mpi.finalize` and the mandatory-result `mpi.comm_world` have no
operands and spell their single `!mpi.retval` or `!mpi.comm` result after their
attribute dictionary. `mpi.error_class` spells its `!mpi.retval` operand type,
and its result has that same type. Ordinary attribute dictionaries remain
queryable in their exact ODS positions. The no-result variants of `mpi.init`
and `mpi.finalize` remain on recovery because the reusable zero-operand shape
requires the colon and result type.

The other 11 operations remain on whole-operation recovery, grouped by the
information omitted or mixed by their concrete headers:

- Communicator queries: `mpi.comm_rank`, `mpi.comm_size`, and `mpi.comm_split`
  put operands in a call-like parenthesized list before `attr-dict`, infer the
  input communicator type, and spell only their optional return code plus
  `i32` or communicator result types.
- Point-to-point and completion: `mpi.send`, `mpi.recv`, `mpi.isend`,
  `mpi.irecv`, and `mpi.wait` mix explicit buffer/integer/request operand types
  with inferred communicator types and optional or mandatory arrows for
  `!mpi.retval` and `!mpi.request` results. The destination/source, tag, and
  receive buffer are all real SSA operands; no variadic or destination-style
  shortcut is applied.
- Collective and synchronization: `mpi.allreduce` has two buffer operands, a
  positional reduction enum, and a communicator operand, but spells only the
  two buffer types and an optional return type. `mpi.barrier` infers its
  communicator input and has an optional return-code arrow.
- Error comparison: `mpi.retval_check` places a positional error-class
  attribute after its return-value operand and spells only the `i1` result.

None of the 15 operations owns regions, successors, or symbol roles. The
`!mpi.retval`, `!mpi.comm`, `!mpi.request`, and `!mpi.status` types and
`#mpi.errclass<...>` attribute work as balanced opaque dialect values without
MPI-specific descriptors. Bare reduction enum tokens such as `MPI_SUM` stay
inside recovered custom operations. The preset adds no MPI type inference,
error checking, communication semantics, or custom verification.

NVGPU defines exactly 24 operations in LLVM 22.1. The preset registers seven
forms whose concrete headers spell every operand and result role needed by the
reusable shapes:

- Matrix multiplication: `nvgpu.mma.sync` exposes its three parenthesized
  operands and complete three-input-to-one-result vector signature.
- Barrier and descriptor primitives: `nvgpu.mbarrier.create` and
  `nvgpu.warpgroup.mma.init.accumulator` have no operands and spell their one
  result type. `nvgpu.tma.fence.descriptor` and the predicate-free form of
  `nvgpu.tma.prefetch.descriptor` each expose one typed descriptor operand and
  no SSA result.
- Warpgroup matrix operations: `nvgpu.warpgroup.generate.descriptor` and
  `nvgpu.warpgroup.mma` spell complete two- and three-input signatures and one
  descriptor or accumulator result.

The other 17 operations remain on whole-operation recovery, grouped by the
structure omitted or repurposed by their concrete headers:

- Indexed and inferred roles: `nvgpu.ldmatrix`, all seven indexed
  `nvgpu.mbarrier.*` operations other than `mbarrier.create`, and
  `nvgpu.tma.create.descriptor` omit index operand types or infer a pointer,
  token, or `i1` result. `nvgpu.tma.async.load` and
  `nvgpu.tma.async.store` use `to` or an arrow to describe a destination while
  producing no SSA result. Treating those trailers as result signatures would
  fabricate results or assign container types to indices.
- Asynchronous tokens: `nvgpu.device_async_copy` infers its token result and
  mixes two bracketed index lists with a destination-style `to` signature;
  `nvgpu.device_async_create_group` infers its token result and
  `nvgpu.device_async_wait` has no type trailer.
- Positional special cases: `nvgpu.mma.sp.sync` omits the metadata operand type
  from its otherwise complete signature, `nvgpu.warpgroup.mma.store` uses `to`
  between two operand types despite having no result, and `nvgpu.rcp` places a
  bare rounding-mode enum in a custom brace clause. The predicated form of the
  registered TMA prefetch also recovers because it spells one descriptor type
  for two operands.

No NVGPU operation owns a region or successor, and `transform.nvgpu.*`
operations belong to the Transform dialect rather than this inventory. NVGPU
types such as device tokens, barrier groups and tokens, tensor-map descriptors,
and warpgroup descriptors and accumulators remain balanced opaque dialect
values. Its enum attributes do likewise when used in generic attribute syntax.
The preset adds no NVGPU verification, type inference, memory effects, or
execution semantics.

NVVM has 185 concrete operations in LLVM 22.1. This inventory comes from
expanding `NVVMOps.td` with LLVM TableGen and selecting concrete `Op` records,
then prefixing their `opName` with `nvvm.`. Counting handwritten `def`
statements is not sufficient: the expansion includes all 32
`nvvm.read.ptx.sreg.envreg0` through `envreg31` operations created by a
`foreach`, as well as records assembled through shared classes. GPU target
attributes and `transform.*` extensions are separate namespaces and are not
part of the 185-operation count.

The preset registers 71 operations in five structurally exact families:

- Result-only registers: 44 non-rangeable forms use
  `attr-dict : type(result)`. This includes clock and timer registers, all 32
  environment registers, lane-mask registers, and the three shared-memory-size
  registers.
- Attribute-only instructions: 16 no-result barriers, fences, group commits,
  cluster controls, exit/breakpoint operations, and allocation-permit release
  forms use only `attr-dict`.
- Same-typed unary intrinsic: `nvvm.rcp.approx.ftz.f` spells its operand and
  shared operand/result type explicitly.
- Typed no-result instructions: `nvvm.bar.warp.sync`,
  `nvvm.cp.async.mbarrier.arrive`, `nvvm.mbarrier.inval`, and
  `nvvm.tcgen05.shift` each spell one SSA operand and its type.
- Complete signatures: `nvvm.ldmatrix`, `nvvm.wmma.mma`,
  `nvvm.mbarrier.arrive.nocomplete`,
  `nvvm.mbarrier.arrive_drop.nocomplete`, `nvvm.mbarrier.test.wait`, and
  `nvvm.tcgen05.mma_smem_desc` spell every operand type and result type in a
  function-type or equivalent arrow signature.

The remaining 114 operations stay on whole-operation recovery. Their gaps are
grouped by the information their assembly omits or gives a positional meaning:

- Thirty-three rangeable special-register operations add an optional
  positional `range` clause to the otherwise reusable result-only form.
- Barrier, memory-barrier, mbarrier, proxy-fence, shuffle, vote, match, redux,
  grid-dependency, and register-control forms use positional scope, action,
  kind, reduction, or predicate fields. Several mbarrier operations also make
  their result arrow or operand list optional. These forms are not registered
  as simpler defaults because that would make valid variants fail under the
  preset.
- The conversion, dot, matrix load/store, and tcgen05 families use mixed type
  lists, type tags in parentheses, inferred operand or result types, or lists
  that omit an SSA operand such as a stride. In particular,
  `nvvm.wmma.load`'s functional type omits its stride operand, so it is not part
  of the complete-signature family.
- Inline PTX, MMA and sparse/block-scale MMA, WGMMA, prefetch, bulk store,
  cluster-launch control, and asynchronous copy/TMA forms combine custom
  delimiters, positional attributes, predicates, or inferred results. Commit
  and fence operations are registered only where their full syntax is the
  attribute dictionary itself.

No NVVM operation in this inventory owns a region, successor, or symbol role.
The dialect's `!nvvm.*` types and `#nvvm.*` attributes already parse as balanced
opaque dialect values and need no descriptors. The preset adds no SM/version
checks, memory effects, intrinsic selection, type inference, or other NVIDIA
target semantics.

OpenMP defines 54 concrete `omp.*` operations in LLVM 22.1. This inventory
comes from the concrete `OpenMP_Op`, `OpenMPTransform_Op`, and
`OpenMPTransformBase_Op` records in `OpenMPOps.td`, including the three loop
transformation operations because they still define `omp.*` operations.
OpenACC/OpenMP common interfaces and helpers, translation support, passes, and
`transform.*` extensions do not add operations to this count.

The preset registers eight conservative default forms. `omp.terminator`,
`omp.taskyield`, and `omp.barrier` are attribute-only, no-operand operations.
`omp.section`, `omp.workshare.loop_wrapper`, `omp.master`, and
`omp.workdistribute` expose one plain region and its explicit block arguments;
forms with an attribute dictionary after that region remain on recovery.
`omp.threadprivate` exposes its one address operand and one result through the
complete `input-type -> result-type` spelling. None of these forms has a
successor or a symbol role. The region operations and control points have no
results, while `omp.threadprivate` has no region.

The remaining 46 operations stay on whole-operation recovery, grouped by the
syntax that prevents a structurally exact registration:

- Inferred-result and loop-transformation forms: `omp.new_cli`,
  `omp.canonical_loop`, `omp.unroll_heuristic`, and `omp.tile`.
- Clause-heavy, custom-bound, keyword-separated, or symbol-bearing region
  forms: `omp.private`, `omp.parallel`, `omp.teams`, `omp.sections`,
  `omp.single`, `omp.workshare`, `omp.loop_nest`, `omp.loop`, `omp.wsloop`,
  `omp.simd`, `omp.distribute`, `omp.task`, `omp.taskloop`, `omp.taskgroup`,
  `omp.target_data`, `omp.target`, `omp.critical`, `omp.ordered.region`,
  `omp.atomic.update`, `omp.atomic.capture`, `omp.declare_mapper`,
  `omp.declare_reduction`, and `omp.masked`. Their region arguments may be
  inferred from private, reduction, map, or loop clauses, and several place
  dictionaries after the final region.
- Parenthesized operand/type clauses, depend/task clauses, mappings, and other
  regionless custom forms: `omp.yield`, `omp.flush`, `omp.map.bounds`,
  `omp.map.info`, `omp.target_enter_data`, `omp.target_exit_data`,
  `omp.target_update`, `omp.critical.declare`, `omp.ordered`, `omp.taskwait`,
  `omp.atomic.read`, `omp.atomic.write`, `omp.cancel`,
  `omp.cancellation_point`, `omp.scan`, `omp.declare_mapper.info`, and
  `omp.allocate_dir`.
- Device allocation forms: `omp.target_allocmem` has handwritten syntax with
  separated type-parameter and shape groups, while `omp.target_freemem` uses
  an unparenthesized mixed operand-type list without an arrow. Treating either
  as a shared or complete function signature would assign false roles.

The `!omp.cli` and `!omp.map_bounds_ty` types and OpenMP attributes remain
balanced opaque dialect values. The preset does not implement OpenMP clause
semantics, private/reduction maps, offload mapping, symbol resolution, type or
block-argument inference, verification, translation, or execution semantics.

PDL defines 15 concrete `pdl.*` operations in LLVM 22.1. This is a core-only
preset: all 15 operations remain on whole-operation recovery. The inventory is
limited to `PDLOps.td`; the separate `pdl_interp.*` execution dialect and
`transform.*` PDL extension are not included.

The gaps are grouped by the information omitted or given a positional role by
their concrete headers:

- Inferred constraint and handle results: `pdl.attribute`, `pdl.operand`,
  `pdl.operands`, `pdl.operation`, `pdl.result`, `pdl.results`, `pdl.type`, and
  `pdl.types` infer their `!pdl.attribute`, `!pdl.value`, `!pdl.operation`,
  `!pdl.type`, or `!pdl.range<...>` result. Their optional type constraints,
  operation names, result-group indices, attribute bindings, and result-type
  handles do not form complete SSA operand/result signatures.
- Native calls and range construction: `pdl.apply_native_constraint` and
  `pdl.apply_native_rewrite` place a positional string name before a typed
  argument list and spell only their optional result types. `pdl.range` has a
  typed operand list but derives the range result type from those operands or
  an explicitly printed range type. These are not complete function
  signatures.
- Rewrite actions: `pdl.erase` has one untyped operation handle, while
  `pdl.replace` selects between an untyped replacement operation and a typed
  replacement-value list after `with`. They produce no SSA results, so treating
  their operand types as result slots or registering them as typed terminators
  would be structurally false.
- Symbols and regions: `pdl.pattern` optionally defines a symbol, carries a
  positional benefit attribute, and owns an isolated single-block region.
  `pdl.rewrite` is the pattern terminator and may combine an optional root
  operation, a named external rewrite with typed arguments, or an inline
  single-block region. A region-only shape would lose those operand, symbol,
  attribute, and terminator roles.

No PDL operation has successors. `pdl.pattern` is the only symbol definition;
PDL's quoted native function and operation names are positional string
attributes, not MLIR symbol references. Its `!pdl.attribute`,
`!pdl.operation`, `!pdl.type`, `!pdl.value`, and `!pdl.range<...>` types remain
balanced opaque dialect values without descriptors. PDL defines no dialect
attributes; namespaced opaque attributes used by generic IR are handled by the
same generic path. The preset does not implement PDL handle inference,
operation-description constraints, rewrite semantics, or ODS interpretation.

PDLInterp defines 39 concrete `pdl_interp.*` operations in LLVM 22.1. This is
a core-only preset: all 39 operations remain on whole-operation recovery. The
inventory is the concrete operation definitions in `PDLInterpOps.td`; the
higher-level `pdl.*` dialect is covered by its separate preset.

Eighteen operations carry CFG successors, and none can use a current reusable
shape without discarding those edges. Nine two-way predicate terminators
(`pdl_interp.apply_constraint`, `pdl_interp.are_equal`, the six
`pdl_interp.check_*` operations, and `pdl_interp.is_not_null`) branch to true
and false destinations. `pdl_interp.branch` and `pdl_interp.record_match` each
have one destination. The six `pdl_interp.switch_*` terminators have a default
destination plus variadic case destinations whose order corresponds to a
positional case-values attribute. `pdl_interp.foreach` has one successor in
addition to its operand and region. Registering any of these with a
no-successor shape would erase real CFG structure.

The remaining gaps are grouped by their semantic roles:

- External calls and inferred handles: `pdl_interp.apply_constraint` and
  `pdl_interp.apply_rewrite` have positional names, variadic typed arguments,
  and variadic handle results. `pdl_interp.create_attribute`,
  `pdl_interp.create_type`, and `pdl_interp.create_types` infer one handle
  result from a positional attribute. `pdl_interp.create_operation` combines
  an operation name, separately segmented operand, attribute-handle, and
  result-type-handle operands, an optional `<inferred>` marker, and one inferred
  operation-handle result. `pdl_interp.create_range` derives its range result
  from a variadic typed input list or an explicitly printed empty-range type.
- Navigation: `pdl_interp.extract` has a positional index, one range operand,
  and one explicitly typed result. The nine `pdl_interp.get_*` operations each
  have one operand and one inferred or explicitly printed result; attribute,
  operand, and result lookup forms also carry required or optional positional
  names or indices. No current shape preserves all of those roles without
  inventing an ordinary function signature.
- Rewrite and termination: `pdl_interp.erase` has one untyped operation-handle
  operand, and `pdl_interp.replace` combines that operand with an optional
  typed replacement list. `pdl_interp.continue` and `pdl_interp.finalize` are
  zero-operand, zero-result terminators. The reusable optional-typed-operands
  shape preserves zero results but accepts operands, so it is not an exact
  registration for either operation.
- Regions and symbols: `pdl_interp.foreach` binds one block argument whose type
  is the element type of its range operand, owns a region terminated by
  `pdl_interp.continue`, and then names its successor. `pdl_interp.func` defines
  a symbol, owns an isolated SSACFG body, and derives entry-block arguments
  from its function signature. `pdl_interp.record_match` uses a nested rewriter
  symbol reference and combines two variadic operand segments with positional
  benefit, location, root-kind, and generated-operation attributes before its
  successor.

The predicate and switch operands are PDL handles; their compared constants,
counts, names, types, case lists, and optional modifiers are positional
attributes rather than SSA values. PDL's `!pdl.attribute`, `!pdl.operation`,
`!pdl.type`, `!pdl.value`, and `!pdl.range<...>` types remain balanced opaque
dialect values, and unknown `#pdl_interp<...>` attributes remain available to
generic quoted IR. The preset adds no PDL type inference, special CFG parser,
ODS interpretation, verifier, rewrite behavior, or interpreter semantics.

Ptr defines 13 operations in LLVM 22.1. The preset registers four default
forms. `ptr.to_ptr`, metadata-free `ptr.from_ptr`, and unmodified `ptr.load`
have one operand and an explicit input-to-result signature. Unflagged
`ptr.ptr_diff` has two same-typed pointer inputs and one explicitly typed
integer or index result. Their ordinary attribute dictionaries are preserved.

Nine operations remain unregistered. `ptr.gather` and `ptr.masked_load` omit
the mask and passthrough operand types from their conversion trailers.
`ptr.store`, `ptr.scatter`, and `ptr.masked_store` have no results and print
types for only a subset of their operands. `ptr.ptr_add` infers its result from
the base and offset shapes, while `ptr.get_metadata` derives a metadata result
from its input type. `ptr.constant` uses a typed dialect attribute and places
its ordinary dictionary before that positional value; the current literal
shape does not accept either the dictionary-free `#ptr.null` default or that
dictionary placement. `ptr.type_offset` has a positional type attribute rather
than a literal attribute. Registering these forms with nearby shapes would
invent missing types or result slots, or misclassify an attribute category.

Optional syntax on otherwise registered operations remains explicit recovery:
the metadata operand on `ptr.from_ptr`; volatile, atomic, synchronization,
ordering, invariant, nontemporal, and alignment clauses on `ptr.load`; and
wrap flags on `ptr.ptr_diff`. These clauses occur before the stable type
boundary and do not match the narrow default shapes. The Ptr operations have
no regions, successors, or symbol roles. `!ptr.*` types and `#ptr.*`
attributes remain balanced opaque dialect values; the preset adds no pointer
type inference, data-layout behavior, memory semantics, or verification.

Quant defines exactly three operations in LLVM 22.1: `quant.dcast`,
`quant.qcast`, and `quant.scast`. All three are registered with their complete
declarative format: one SSA operand, an optional ordinary attribute dictionary,
and explicit operand and result types separated by `to`. `quant.dcast` converts
a quantized input to its expressed floating-point type, `quant.qcast` converts
an expressed floating-point input to a quantized result, and `quant.scast`
converts between a quantized value and its integer storage representation.
Their scalar, ranked-tensor, and unranked-tensor spellings use the same header.

No Quant operation infers a result type, owns a region or successor, or has a
symbol role. Statistics operations found in older descriptions of the dialect
are not part of the pinned LLVM 22.1 `QuantOps.td` inventory. Quantized types
such as `!quant.uniform<...>` and namespaced `#quant.*` attributes remain
balanced opaque values and need no dialect descriptors. The preset preserves
the structural operand/result signatures and ordinary dictionary attributes;
it does not implement Quant's expressed/storage type checks, shape checks,
per-axis integrity rules, folding, or quantization semantics.

ROCDL has 323 concrete `rocdl.*` operations in LLVM 22.1. The inventory was
produced by expanding `ROCDLOps.td` with LLVM TableGen's JSON backend and
selecting concrete `Op` records whose dialect is `ROCDL_Dialect`. This matters
for the generated MFMA, SMFMAC, WMMA, scaled-WMMA, conversion, and memory
families: counting visible `def` statements is not an exact inventory.
ROCDL target attributes and GPU/Transform operations that consume them are
not operations in this namespace and are outside the count.

The preset registers 125 operations in assembly families whose complete SSA
operand and result types are spelled by the custom form:

- All 47 MFMA and 28 SMFMAC operations use a variadic operand list and a full
  functional type. All 38 WMMA operations do the same for three or five SSA
  operands. Named WMMA parameters such as sign, clamp, reuse, format, scale
  type, and `opsel` are attributes in the ordinary dictionary; they are not
  counted as SSA operands. MFMA's variadic `$args`, including intrinsic
  immediate arguments represented by SSA constants, remain operands.
- `rocdl.mbcnt.lo`, `rocdl.mbcnt.hi`, `rocdl.ds_swizzle`,
  `rocdl.ds_bpermute`, and `rocdl.readlane` spell complete two-input function
  types. `rocdl.readfirstlane` has one same-typed input and result.
- The four `rocdl.ds.read.tr*.b*` operations spell an unqualified pointer type
  and a result type on opposite sides of an arrow. `rocdl.barrier` and
  `rocdl.s.barrier` are exact zero-operand, zero-result forms.

The remaining 198 operations stay on whole-operation recovery, grouped by the
reason a nearby shape would be structurally dishonest:

- Sixteen special-register and dimension operations have an optional
  positional `range` attribute. The preset does not register only their
  dictionary-and-result suffix because that would reject a valid range.
- Twenty-three synchronization and scheduling operations use positional
  immediate attributes such as `id =`, `member_cnt =`, count, mask, priority,
  bitfield, size, group, scope, or variant. Some also use qualified pointer
  types or an arrow without a complete SSA function signature.
- Forty-two raw-buffer, pointer-buffer, LDS/tensor load-store, prefetch, and
  asynchronous memory operations use custom C++ assembly, qualified pointer
  types, cache-policy attributes, or signatures that omit or infer some SSA
  operand or result types.
- One hundred nine conversion, permutation, DPP, ballot, and median operations
  mix SSA values with positional selector, scale, seed, destination, and
  modifier attributes, while spelling only a result type or another partial
  signature. These positional fields are not modeled as operands.
- Eight scalar math operations use `qualified(type(...))` on both sides of a
  one-off arrow form. The current preset does not broaden the format system
  solely for this family.

No concrete ROCDL operation owns a region or successor, and none defines or
uses an MLIR symbol. LLVM pointer/vector types and ROCDL/LLVM attributes remain
balanced opaque values. The preset adds no target-availability checks,
intrinsic verification, immediate validation, type inference, memory effects,
or execution semantics.

Shard defines 22 operations in LLVM 22.1, rather than the earlier estimate of
21. The preset registers `shard.get_sharding`, whose one ranked-tensor operand,
one `!shard.sharding` result, and explicit input-to-result conversion signature
fit the unary shape exactly. Its ordinary attribute dictionary remains
queryable. The operation has no regions, successors, or symbol roles. The
preset adds no Shard-specific type inference or same-type verification.

The other 21 operations remain on whole-operation recovery. `shard.grid`
defines a symbol using a custom dimension list. `shard.grid_shape`, the two
process-index operations, `shard.neighbors_linear_indices`, and all 12
collectives use positional grid symbols; their forms also combine inferred
index results, grid-axis arrays, tensor-axis or reduction clauses, dynamic
root/source/destination index lists, and differing input/result tensor types.
`shard.sharding` and `shard.shard_shape` use mixed static/dynamic index lists;
`shard.shard` has an optional unit clause and a same-input/result-type
relationship; `shard.update_halo` combines a destination operand, a grid
symbol, split axes, and optional mixed halo sizes while spelling only the
result type. Registering these forms with the broad clause shape would leave
grid references without symbol-use semantics and would infer incorrect types
or result slots for several families.

No Shard operation owns a region or successor. `shard.grid` is the sole symbol
definition; every operation with a `grid` attribute is a symbol user. The
`!shard.sharding` type, `#shard.partial` reduction attribute, and Shard
grid-axis attributes remain opaque balanced namespaced values. The preset does
not implement grid-symbol resolution, collective semantics, destination-style
semantics, or Shard-specific verification.

MLProgram defines 11 operations in LLVM 22.1. This is a core-only preset: all
11 operations remain on whole-operation recovery, grouped by the structure
that the current registry cannot represent faithfully:

- Symbol definitions and regions: `ml_program.func`, `ml_program.subgraph`, and
  `ml_program.global`. The two callable operations differ in region kind, and
  the global combines a symbol with an optional typed program-variable
  initializer. A name-only shape cannot supply their symbol or region metadata.
- Symbol users and program-variable access: `ml_program.global_load`,
  `ml_program.global_load_const`, `ml_program.global_store`,
  `ml_program.global_load_graph`, and `ml_program.global_store_graph`. Their
  leading global references are attributes rather than SSA operands, and the
  graph forms add custom token-ordering clauses with inferred token types.
- Inferred result type: `ml_program.token` spells neither its
  `!ml_program.token` result type nor a type signature.
- Terminators: `ml_program.output` and `ml_program.return` have ordinary
  optional typed SSA operands, but their attribute dictionary precedes that
  optional clause. The reusable typed-terminator shape accepts its dictionary
  after the operands, so registering it would reject valid attributed forms.

None of the 11 operations has successors. Only `ml_program.func` and
`ml_program.subgraph` own regions; their region kinds are SSACFG and Graph,
respectively. The declarative global and access forms put attribute dictionaries
after their final typed clause, while the terminators put them before their
optional operands. The `!ml_program.token` type and `#ml_program.extern`
attribute remain opaque balanced dialect values. The preset does not implement
global-symbol resolution, callable semantics, program-variable type inference,
token ordering, or MLProgram verification.

`region_clauses` is used for operations such as `scf.for`, `scf.if`,
`tosa.while_loop`, and `linalg.generic`. Their operation regions, explicit block
labels, and typed block arguments use Zirium's ordinary semantic region model.
Header bindings written as `%argument = %initial` are attached to implicit entry
blocks as opaque-typed arguments when the custom syntax does not spell their
types locally. This preserves definition/use structure without claiming dialect
type inference.

SCF defines 12 operations in LLVM 22.1. The preset structurally registers 11.
`scf.condition` remains on whole-operation recovery because its fixed
parenthesized `i1` condition is followed by an optional typed forwarding-operand
clause. Treating its trailing types as results would be incorrect: the operation
has no results. `scf.reduce.return` and `scf.yield` use the typed-terminator shape,
so their typed SSA values are operands and both operations have zero results.
The remaining registered operations preserve their SSA operands, result arity,
and owned regions. None of the SCF operations has successors.

## Python

Pass the registry when parsing. The parsed file retains it, and its semantic
documents use it for verification, editing, and custom printing.

Load a bundled preset by name:

```python
registry = zirium.DialectRegistry.from_name("stablehlo")
```

```python
import zirium

registry = zirium.DialectRegistry.proving().extend_operation_shapes({
    "vendor.function": zirium.OperationShape.FUNC_LIKE,
    "vendor.invoke": zirium.OperationShape.CALL_LIKE,
})
source = '''module {
  vendor.function @declaration()
}'''
parsed = zirium.parse_text(source, registry=registry)
assert parsed.diagnostics == []
lowered = parsed.lower_strict("hybrid")
assert lowered.document is not None
operation = lowered.document.operation_table("vendor.function").operation(0)
assert operation.symbol_name == "declaration"
```

`with_operation_shapes(...)` starts with core operations.
`existing_registry.extend_operation_shapes(...)` preserves the existing
registry. Both accept Python mappings and return a new registry.

`FUNC_LIKE` and `CALL_LIKE` provide the existing symbol-oriented forms.
`BINARY_OPERANDS` accepts two SSA operands followed by either one shared type or
a function type. `OPTIONAL_TYPED_OPERANDS` accepts a variadic operand list with
a matching optional type list. `OPERAND_CLAUSES` captures SSA operands and
simple named attributes around otherwise opaque fixed clauses, followed by a
shared, conversion, or function-type signature. It is useful for inspection;
it does not interpret the clauses' dialect-specific meaning. `REGION_CLAUSES`
adds one or more parsed operation regions and entry-block header bindings to
that structural model.

Shapes supply parsing and lowering conventions. They do not define a vendor
operation's verifier, symbol-table rules, or custom printer. In particular,
`custom_bytes()` prints caller-defined shapes in generic form. Registering a
func-like shape does not make it equivalent to `func.func` for every semantic
analysis.

`lower_strict()` rejects lowering errors. It does not replace
`document.verify_semantics()`. Best-effort lowering can return an incomplete
document; inspect its diagnostics and `semantically_complete` before choosing
an output or editing path.

`custom_bytes()` prefers the built-in assembly printers. It falls back to
generic form for unsupported structure, properties, locations, typed constant
attributes, and string-valued function or module names that the current custom
printer cannot represent faithfully. Use `canonical_bytes()` for deterministic
generic output and `preserving_bytes()` for eligible source-preserving edits.
Validation failures from `write_custom()` and `write_canonical()` raise
`ValueError`; file creation, write, and flush failures raise `OSError`.

## Rust

Use the same registry for parsing, lowering, verification, editing, and
`print_with_registry`. `ParsedFile` does not retain an owned registry. Text
edits on registered syntax need `apply_text_edits_with_registry`; the shorter
`apply_text_edits` method reparses with the empty registry.

## JSON configuration and Pydantic models

The CLI and Python accept the same complete registry configuration:

```json
{
  "presets": [],
  "builtins": ["builtin.module", "arith.constant", "arith.addi"],
  "operation_shapes": [
    {"name": "vendor.function", "shape": "func_like"},
    {"name": "vendor.invoke", "shape": "call_like"}
  ],
  "operation_formats": [
    {"name": "vendor.widen", "format": "$operands attr-dict `:` type($operands) `into` type($results)"}
  ]
}
```

`builtins` and `operation_shapes` are required and may be empty. `presets` and
`operation_formats` default to empty. `presets` adds bundled registries by
name. `builtins` selects operations from the declarative catalog.
`operation_shapes` assigns exact names to a supported grammar: `func_like`,
`call_like`, `binary_operands`, or `optional_typed_operands`.
`unary_operand`, `variadic_operands`, and `literal_attribute` cover the smaller
expression forms. `operand_clauses` accepts fixed and named clauses around SSA
operands before a trailing type signature. `region_clauses` additionally parses
operation regions and their block arguments.

`operation_formats` describes the order of operands or a literal value, an
optional attribute dictionary, fixed tokens, and type captures. Operand forms
accept `to` or `into` as the result-type separator; the JSON example above uses
`into`. The supported literals are `:`, `to`, and `into`; other ODS format
literals are rejected. The registry checks each description when it is
constructed. The typed-literal form is
``$value `:` type($value) attr-dict `:` type($result)``.

This configuration replaces the caller's default registry.

For the bundled StableHLO subset:

```json
{
  "presets": ["stablehlo"],
  "builtins": [],
  "operation_shapes": []
}
```

Python accepts an ordinary JSON-compatible dictionary:

```python
config = {
    "presets": [],
    "builtins": ["builtin.module"],
    "operation_shapes": [
        {"name": "vendor.function", "shape": "func_like"},
    ],
}
registry = zirium.DialectRegistry.from_config(config)
parsed = zirium.parse_text("module { vendor.function @f() }", registry=registry)
```

Or use the Pydantic definitions:

```python
from pathlib import Path

config = zirium.RegistryConfig(
    builtins=["builtin.module"],
    operation_shapes=[
        zirium.OperationShapeConfig(name="vendor.function", shape="func_like"),
    ],
)
registry = zirium.DialectRegistry.from_config(config)
Path("registry.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
registry = zirium.DialectRegistry.from_file("registry.json")
schema = zirium.RegistryConfig.model_json_schema()
```

The Pydantic models check the structure without coercing input types. The shared
Rust builder checks registered names, duplicates, and conflicts for both files
and Python data. Unknown fields, invalid shapes, and invalid registrations raise
`ValueError`; file I/O failures raise `OSError`. Non-JSON-compatible objects in
a dictionary raise `TypeError`. Loading does not retain the input dictionary.

Combine several configurations by passing additional arguments:

```python
registry = zirium.DialectRegistry.from_file("common.json", "vendor.json")
registry = zirium.DialectRegistry.from_config(common_config, vendor_config)
```

Built-ins and identical shape definitions shared across configurations are
included once. Conflicting shapes fail, as does a custom shape that collides
with a built-in selected by another configuration. Duplicates within one
configuration are errors. No file overrides another based on order.

## The binary

Build or install the binary separately with Cargo:

```sh
cargo install --path crates/zirium
zirium --registry examples/cli/registry.json \
  'filter(op("vendor.function")) | count' examples/cli/registered-shapes.mlir
zirium --registry common.json --registry vendor.json -f inspect.zirium input.mlir
```

The Python wheel provides the extension and Python API; it does not install
this binary. With no `--registry` flags the binary uses the proving registry.
With one or more flags it uses their combined configuration. Registry files
are read as UTF-8 JSON through Serde before any MLIR input is read. Relative
paths resolve against the working directory, and stdin remains reserved for
MLIR. Registry failures produce no query output.

Place `--registry` before an inline query. With `-f`/`--program-file`, registry
options may appear before or after the program-file pair, until the first
input path. Remaining arguments are input paths. `--` ends option parsing.

The CLI can select or count recovered unknown custom operations. It rejects
other syntax errors and semantic lowering diagnostics. Semantic mutations
require a complete document. Output uses the selected-fragment printer, including
when the selection contains the whole input. `closure` additionally
requires registered reference semantics; configuring a func-like or call-like
shape alone does not supply vendor dependency semantics. See
the [query language reference](query-language.md) for the query syntax and
selected-fragment output contract, or the [CLI examples](cli-examples.md) for
worked commands.

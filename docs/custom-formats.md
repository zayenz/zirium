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
| SCF preset | Core plus all 12 SCF operations, including structured regions and loop header bindings. |
| Linalg preset | Core plus 97 core, structured, and generated named Linalg operations. |
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
`arm_neon`, `arm_sme`, `arm_sve`, `async`, `bufferization`, `cf`, `complex`, `dlti`, `emitc`, `func`, and `gpu` presets were checked against
LLVM 22.1.0.
TOSA registers 93 of the 94 operations defined by its main, utility, and shape
operation files; `tosa.variable` remains on the generic recovery path because
its custom symbol/type form has no reusable structural signature. SCF registers
all 12 operations. Linalg registers its 16 core/structured operations and 81
generated named operations. Tensor-result named Linalg forms expose their
trailing result types; buffer forms without a result signature remain usable
through recovery where their custom spelling has no safe structural boundary.

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

`region_clauses` is used for operations such as `scf.for`, `scf.if`,
`tosa.while_loop`, and `linalg.generic`. Their operation regions, explicit block
labels, and typed block arguments use Zirium's ordinary semantic region model.
Header bindings written as `%argument = %initial` are attached to implicit entry
blocks as opaque-typed arguments when the custom syntax does not spell their
types locally. This preserves definition/use structure without claiming dialect
type inference.

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

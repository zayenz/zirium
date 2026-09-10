!word = type i32
!pair = type tuple<!word, !word>
#answer = 42 : i32
#meta = {z = #answer, a = ["x", @root::@leaf]}
%result:2 = "test.values"() {first = #meta, second = #meta} : () -> (!pair, !pair) loc("core.mlir":7:4)

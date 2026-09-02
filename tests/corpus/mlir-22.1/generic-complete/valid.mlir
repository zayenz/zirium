#location_alias = loc("aliased")
%pair:2, %single = "test.results"() : () -> (i32, f32, index)
"test.uses"(%pair#0, %pair#1, %single) : (i32, f32, index) -> ()
"test.properties"() <{inherent = 7}> {discardable = "yes"} : () -> ()
"test.regions"() ({
  "test.implicit"() : () -> ()
  "test.successors"(%single) [^next : (%single : index, %single : index), ^empty : (), ^exit] : (index) -> ()
  ^next(%arg0 : index loc("argument"), %arg1 : index):
    "test.in_block"(%arg0) : (index) -> () loc("operation")
  ^empty:
    "test.empty"() : () -> ()
  ^exit:
    "test.exit"() : () -> ()
}, {
  ^other:
    "test.other"() : () -> () loc(#location_alias)
}) : () -> () loc(callsite("callee" at "caller"))

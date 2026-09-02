#loc_alias = loc("aliased")
#loc_chain = #loc_alias
#bad_loc = 42
!type_alias = type i32
%ok = "test.location_alias"() {value = loc(#loc_chain)} : () -> i32 loc(#loc_alias)
"test.location_nested_aliases"() {value = loc("outer"(#loc_alias))} : () -> () loc(fused<"pass">[#loc_alias, callsite(#loc_chain at "caller")])
#cycle_a = #cycle_b
#cycle_b = #cycle_a
%bad = "test.location_bad"() {wrong = loc(#bad_loc), missing = loc(#missing_loc), wrong_type = loc(#type_alias)} : () -> tensor<2xf32, #missing_encoding> loc(fused<"pass">["ok", malformed, loc(unknown)])
"test.location_nested_bad"() : () -> () loc(fused<"pass">[#cycle_a, callsite(#bad_loc at "caller"), "outer"(#missing_loc)])
%dups = "test.duplicate_aggregate"() {value = {a = 1, a = 2}} : () -> i32

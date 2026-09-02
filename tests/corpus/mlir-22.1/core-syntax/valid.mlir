"core.values"() {
  integer = -42 : si32,
  hexadecimal = 0x2A : ui64,
  floating = 2. : f32,
  hexadecimal_float = 0x7FC00000 : f32,
  index_value = 8 : index,
  string = "string attribute",
  typed_string = "s" : i32,
  type = i32,
  attribute_alias = #attr_alias,
  symbol = @root::@leaf,
  array = [10, i32, "string attribute"],
  dictionary = {nested = 10, "quoted name" = "value"},
  unknown_location = loc(unknown),
  file_location = loc("mysource.cc":10:8),
  name_location = loc("foo"),
  callsite_location = loc(callsite("foo" at "mysource.cc":10:8)),
  fused_location = loc(fused<"myPass">["foo", "mysource.cc":10:8])
} : (i1, si16, ui32, f64, index, !type_alias) -> !type_alias

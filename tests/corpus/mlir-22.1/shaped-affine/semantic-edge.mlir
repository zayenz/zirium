!ty = i32
#a = #b
#b = affine_map<(d0) -> (d0 + 1)>
#set_a = #set_b
#set_b = affine_set<(d0) : (d0 >= 0)>
#cycle_a = #cycle_b
#cycle_b = #cycle_a
#wrong = !ty
%0 = "affine.semantic.edge"() {
  map = #a,
  set = #set_a,
  empty = affine_map<>,
  empty_set = affine_set<>,
  huge = affine_map<(d0) -> (999999999999999999999999999999999999999999)>,
  cycle = #cycle_a,
  wrong = #wrong,
  unresolved = #missing,
  incomplete = affine_map<
} : () -> memref<2xf32, #a>

"affine.semantic.bad"() {
  arity = affine_map<(d0) -> (d1 + 1)>,
  operator = affine_map<(d0) -> (d0 floordiv)>,
  constraint = affine_set<(d0) : (d0 > 0)>,
  nested = [affine_map<(d0) -> (d0 + )>]
} : () -> ()

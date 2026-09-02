"shaped.affine"() {
  map = affine_map<(d0, d1)[s0] -> (d0 + s0 * 2, d1 floordiv 4, (d0 + s0) * 2, - (d0 + d1))>,
  reserved_map = affine_map<(loc) -> (loc)>,
  set = affine_set<(d0)[s0] : (d0 >= 0, d0 - s0 == 0)>
} : (tuple<i32, f32>, tensor<2x?xf32, #encoding>, vector<[4]x8xf32>, memref<2x3xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>, memref<2xf32, strided<[1], offset: ?>, #space>, memref<2xf32, #vendor.layout<"x">, 2>) -> tensor<*xf32>

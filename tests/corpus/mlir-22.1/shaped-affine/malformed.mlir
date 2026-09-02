"affine.bad"() {map = affine_map<(d0)[s0] -> (-(d0 + ), s0 * 2)>, set = affine_set<(d0) : (d0 > 0)>} : () -> ()
"shaped.bad.suffix"() : (tensor<2xf32, >, memref<2xf32, strided<[1], offset: ?, #space>) -> ()
"affine.after"() : (tensor<2x?xf32>) -> vector<4xi32>

#map_alias = affine_map<(d0, d1)[s0] -> (d0 + s0 * 2, d1 floordiv 4)>
#set_alias = affine_set<(d0)[s0] : (d0 >= 0, d0 - s0 == 0)>
%0 = "shaped.affine.semantic"() {
  map = affine_map<(d0, d1)[s0] -> (d0 + s0 * 2, d1 floordiv 4, (d0 + s0) * 2, - (d0 + d1))>,
  set = affine_set<(d0)[s0] : (d0 >= 0, d0 - s0 == 0)>,
  map_alias = #map_alias,
  set_alias = #set_alias,
  nested = [#map_alias, [affine_set<(d0) : (d0 <= 4)>]],
  same_map = affine_map<(d0, d1)[s0] -> (d0 + s0 * 2, d1 floordiv 4)>
} : () -> memref<2x3xf32, #map_alias>

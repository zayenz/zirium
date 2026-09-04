#map_alias = affine_map<(d0)[s0] -> (d0 + s0 * 2)>
%0 = "shaped.affine.semantic"() {map = #map_alias} : () -> memref<2x3xf32, #map_alias>

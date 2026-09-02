#layout = 7
#space = 4
!space_type = type i32
%valid = "test.memref.valid"() : () -> memref<4xf32, #layout, #space>
%invalid:3 = "test.memref.invalid"() : () -> (memref<4xf32, #missing_layout, #missing_space>, memref<4xf32, #layout, !space_type>, memref<4xf32, strided<[#missing_nested], offset: 0>, 3>)

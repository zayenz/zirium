module {
  %renamed = arith.constant 8 : i32
  %extra = arith.constant 1 : i32
  %sum = arith.addi %renamed, %extra : i32
}

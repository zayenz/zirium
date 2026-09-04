module {
  %lhs = arith.constant 6 : i32
  %rhs = arith.constant 7 : i32
  %sum = arith.addi %lhs, %rhs : i32
  %product = "arith.muli"(%sum, %rhs) {analysis.tag = "old"} : (i32, i32) -> i32
}

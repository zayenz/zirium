%lhs = "arith.constant"() {value = 1 : i32} : () -> i32
%rhs = "arith.constant"() {value = 2 : i32} : () -> i32
%sum = "arith.addi"(%lhs, %rhs) : (i32, i32) -> i32
"test.consume"(%sum) : (i32) -> ()

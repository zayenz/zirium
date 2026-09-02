builtin.module @root attributes {tag = "module"} {
  func.func private @callee() -> i32 attributes {no_inline} {
    %value = arith.constant 7 : i32
    func.return %value : i32
  }
  func.func @caller() -> i32 {
    %condition = arith.constant 1 : i1
    cf.cond_br %condition, ^left, ^right {branch_weights = dense<[1, 2]> : vector<2xi32>}
  ^left:
    %left_value = func.call @callee() : () -> i32
    cf.br ^exit(%left_value : i32)
  ^right:
    %right_value = arith.constant 9 : i32
    cf.br ^exit(%right_value : i32)
  ^exit(%result: i32):
    func.return %result : i32
  }
}

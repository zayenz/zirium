module {
  func.func @callee() -> i32 {
    %value = arith.constant 7 : i32
    func.return %value : i32
  }
}

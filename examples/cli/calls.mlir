module {
  func.func @answer() -> i32 {
    %value = arith.constant 42 : i32
    func.return %value : i32
  }
  func.func @unrelated() {
    func.return
  }
  func.func @caller() -> i32 {
    %result = func.call @answer() : () -> i32
    func.return %result : i32
  }
}

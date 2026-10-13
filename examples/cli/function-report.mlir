module {
  func.func @double(%x: i32) -> i32 {
    %0 = arith.addi %x, %x : i32
    func.return %0 : i32
  }
  func.func @mix(%x: i32, %y: i32) -> i32 {
    %0 = "arith.muli"(%x, %y) : (i32, i32) -> i32
    %1 = arith.addi %0, %x : i32
    func.return %1 : i32
  }
  func.func @plain(%x: i32) -> i32 {
    func.return %x : i32
  }
}

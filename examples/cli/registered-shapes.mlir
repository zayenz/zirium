module {
  vendor.function @identity(%arg: i32) -> i32 {
    "vendor.yield"(%arg) : (i32) -> ()
  }
  %value = arith.constant 7 : i32
  %result = vendor.invoke @identity(%value) : (i32) -> i32
  "vendor.observe"(%result) : (i32) -> ()
}

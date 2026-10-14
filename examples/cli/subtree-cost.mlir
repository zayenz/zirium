"example.root"() ({
  %left = "example.left"() {cost = 3 : i64} : () -> i32
  "example.right"(%left) {cost = 4 : i64} : (i32) -> ()
}) : () -> ()

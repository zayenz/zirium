"test.first"() ({
  %same = "test.value"() : () -> (i32)
  "test.use"(%same) : (i32) -> ()
}) : () -> ()
"test.second"() ({
  %same = "test.value"() : () -> (i32)
  "test.use"(%same) : (i32) -> ()
}) : () -> ()

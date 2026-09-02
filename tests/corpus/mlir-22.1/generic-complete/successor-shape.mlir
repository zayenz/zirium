"test.cfg"() ({
  %value = "test.value"() : () -> (index)
  "test.branch"(%value) [^dest : (%value : index), ^dest : ()] : (index) -> ()
  ^dest(%argument : i32):
    "test.return"() : () -> ()
}) : () -> ()

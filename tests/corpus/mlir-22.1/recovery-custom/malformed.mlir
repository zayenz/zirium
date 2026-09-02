test.unknown %arg {nested = [1, 2]}
"test.after_unknown"() : () -> ()
"test.bad_array"() {value = [1, 2 } : () -> ()
"test.after_array"() : () -> ()
"test.outer"() ({
  test.inner %arg (nested)
  "test.after_inner"() : () -> ()
}) : () -> ()

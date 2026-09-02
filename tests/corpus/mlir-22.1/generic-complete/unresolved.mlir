%pair:2 = "test.results"() : () -> (i32, i32)
"test.bad_use"(%pair#2) : (i32) -> ()
"test.bad_successor"() [^missing] ({
  ^known:
    "test.return"() : () -> ()
}) : () -> ()

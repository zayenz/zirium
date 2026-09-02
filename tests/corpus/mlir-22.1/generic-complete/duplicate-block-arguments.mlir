"test.duplicate_block_arguments"() ({
  ^entry(%same : i32, %same : i32):
    "test.use"(%same) : (i32) -> ()
}) : () -> ()

builtin.module @broken {
  func.func @bad() {
    cf.cond_br, ^left, ^right
  }
}
"after.recovery"() : () -> ()

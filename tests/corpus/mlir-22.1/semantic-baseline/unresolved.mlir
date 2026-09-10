"builtin.module"() ({
  %0 = "vendor.make"() { tag = #vendor.tag<"x"> } : () -> !vendor.token<"x">
  "vendor.consume"(%missing) : (!vendor.token<"x">) -> ()
}) : () -> ()

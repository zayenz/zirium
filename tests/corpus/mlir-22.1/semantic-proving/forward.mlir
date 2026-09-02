"builtin.module"() ({
  "vendor.consume"(%0) : (!vendor.token<"x">) -> ()
  %0 = "vendor.make"() { tag = #vendor.tag<"x"> } : () -> !vendor.token<"x">
}) : () -> ()

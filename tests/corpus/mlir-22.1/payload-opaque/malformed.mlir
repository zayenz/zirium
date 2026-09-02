"payload.bad"() {value = dense<[1, 2} : tensor<2xi32>} : () -> ()
"payload.after"() : () -> ()
"opaque.bad"() {value = #vendor.attr<[1, 2}>} : () -> ()
"opaque.after"() : () -> ()

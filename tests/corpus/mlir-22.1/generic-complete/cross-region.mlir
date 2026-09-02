"test.cross_region"() ({
  ^from:
    "test.jump"() [^to] : () -> ()
}, {
  ^to:
    "test.return"() : () -> ()
}) : () -> ()

%a = arith.constant 1 {tag = "left"} : i32
%b = arith.constant 2 : i32
%sum = arith.addi %a, %b overflow<nsw> {tag = "sum"} : i32
"func.func"() ({
^entry:
  cf.br ^exit {tag = "edge"}
^exit:
  func.return {tag = "return"}
}) : () -> ()

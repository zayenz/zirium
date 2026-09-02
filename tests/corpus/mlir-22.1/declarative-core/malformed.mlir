%missing = arith.constant : i32
%short = arith.addi %missing : i32
cf.br (%missing : i32)
func.return %missing :

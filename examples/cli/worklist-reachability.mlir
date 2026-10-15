%start = "graph.start"() : () -> i32
%left = "graph.node"(%start) : (i32) -> i32
%right = "graph.node"(%start) : (i32) -> i32
"graph.end"(%left, %right, %left) : (i32, i32, i32) -> ()
"graph.disconnected"() : () -> ()

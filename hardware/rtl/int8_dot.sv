module int8_dot (
    input  logic [31:0]     weights,
    input  logic [ 3:0][7:0] activations,
    output logic [31:0]     sum
);

  always_comb begin
    logic signed [31:0] total;
    total = '0;
    for (int index = 0; index < 4; index++)
      total = total + 32'($signed(weights[8*index+:8]) * $signed(activations[index]));
    sum = $unsigned(total);
  end

endmodule

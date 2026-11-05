module ternary_dot
  import tritium_package::*;
(
    input  logic [31:0]      weights,
    input  logic [19:0][7:0] activations,
    output logic [31:0]      sum
);

  `include "trit_table.svh"

  always_comb begin
    logic signed [31:0] total;
    total = '0;
    for (int byte_index = 0; byte_index < 4; byte_index++) begin
      logic [9:0] digits;
      digits = trit_digits(weights[8*byte_index+:8]);
      for (int trit_index = 0; trit_index < 5; trit_index++) begin
        logic signed [31:0] activation;
        activation = 32'($signed(activations[5*byte_index+trit_index]));
        unique case (digits[2*trit_index+:2])
          2'b01: total = total + activation;
          2'b11: total = total - activation;
          default: ;
        endcase
      end
    end
    sum = $unsigned(total);
  end

endmodule

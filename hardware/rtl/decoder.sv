module decoder
  import tritium_package::*;
(
    input  logic     [ 5:0] opcode,
    output decoded_t        decoded
);

  always_comb begin
    decoded = '0;
    decoded.is_valid = 1'b1;
    unique case (opcode)
      OPCODE_EXIT: decoded.kind = KIND_EXIT;
      default: decoded.is_valid = 1'b0;
    endcase
  end

endmodule

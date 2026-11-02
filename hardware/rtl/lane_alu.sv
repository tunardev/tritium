module lane_alu
  import tritium_package::*;
(
    input  logic [ 4:0] operation,
    input  logic [31:0] left,
    input  logic [31:0] right,
    input  logic [31:0] destination,
    output logic [31:0] result
);

  logic [4:0] shift;
  logic is_less;
  logic signed [31:0] product;

  assign shift = right[4:0];
  assign product = $signed(left[15:0]) * $signed(right[15:0]);
  assign is_less = $signed(left) < $signed(right);

  always_comb begin
    unique case (operation)
      FUNCTION_ADD: result = left + right;
      FUNCTION_SUBTRACT: result = left - right;
      FUNCTION_AND: result = left & right;
      FUNCTION_OR: result = left | right;
      FUNCTION_XOR: result = left ^ right;
      FUNCTION_SHIFT_LEFT: result = left << shift;
      FUNCTION_SHIFT_RIGHT_LOGICAL: result = left >> shift;
      FUNCTION_SHIFT_RIGHT_ARITHMETIC: result = $unsigned($signed(left) >>> shift);
      FUNCTION_SET_LESS_THAN: result = {31'b0, is_less};
      FUNCTION_SET_LESS_THAN_UNSIGNED: result = {31'b0, left < right};
      FUNCTION_MINIMUM: result = is_less ? left : right;
      FUNCTION_MAXIMUM: result = is_less ? right : left;
      FUNCTION_MULTIPLY_HALFWORDS: result = $unsigned(product);
      FUNCTION_SELECT: result = destination != 0 ? left : right;
      FUNCTION_SATURATE_INT8: begin
        if ($signed(left) > 32'sd127) result = 32'd127;
        else if ($signed(left) < -32'sd127) result = -32'd127;
        else result = left;
      end
      default: result = '0;
    endcase
  end

endmodule

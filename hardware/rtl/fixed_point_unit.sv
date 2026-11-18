module fixed_point_unit
  import tritium_package::*;
(
    input  logic        clock,
    input  logic        reset,
    input  logic        start,
    input  logic [ 4:0] operation,
    input  logic [31:0] left,
    input  logic [31:0] right,
    output logic        done,
    output logic [31:0] result
);

  `include "exp2_table.svh"
  `include "rsqrt_table.svh"

  logic [31:0] exp2_result;
  logic [31:0] rsqrt_result;
  logic [4:0] exponent;
  logic [4:0] even_exponent;
  logic [31:0] mantissa;

  logic is_finishing;
  logic [7:0] exp2_index;
  logic signed [31:0] exp2_shift;
  logic is_exp2_saturated;
  logic [7:0] rsqrt_index;
  logic [4:0] rsqrt_exponent;
  logic is_rsqrt_zero;

  logic is_multiplying;
  logic [5:0] step;
  logic [63:0] multiplicand;
  logic [31:0] multiplier;
  logic [63:0] product;
  logic is_negative;
  logic is_rounding;
  logic [31:0] rounded_result;

  always_comb begin
    exponent = '0;
    for (int bit_index = 0; bit_index < 32; bit_index++) if (left[bit_index]) exponent = 5'(bit_index);
    even_exponent = {exponent[4:1], 1'b0};
    if (even_exponent >= 6) mantissa = left >> (even_exponent - 6);
    else mantissa = left << (6 - even_exponent);
  end

  always_comb begin
    logic [31:0] entry;
    entry = 32'(exp2_entry(exp2_index));
    if (is_exp2_saturated) exp2_result = 32'd65536;
    else if (exp2_shift > 16) exp2_result = '0;
    else exp2_result = (entry + (32'd1 << (exp2_shift - 1))) >> exp2_shift;
  end

  always_comb begin
    logic [31:0] estimate;
    logic [4:0] shift;
    estimate = 32'(rsqrt_entry(rsqrt_index));
    shift = (rsqrt_exponent - 16) / 2;
    if (is_rsqrt_zero) rsqrt_result = 32'h7FFF_FFFF;
    else if (rsqrt_exponent <= 16) rsqrt_result = estimate << ((16 - rsqrt_exponent) / 2);
    else rsqrt_result = (estimate + (32'd1 << (shift - 1))) >> shift;
  end

  always_comb begin
    logic signed [63:0] signed_product;
    logic signed [63:0] rounded;
    signed_product = is_negative ? -$signed(product) : $signed(product);
    rounded = (signed_product + 64'sd32768) >>> 16;
    if (rounded > 64'sd2147483647) rounded_result = 32'h7FFF_FFFF;
    else if (rounded < -64'sd2147483648) rounded_result = 32'h8000_0000;
    else rounded_result = rounded[31:0];
  end

  always_ff @(posedge clock) begin
    done <= 1'b0;
    if (reset) begin
      is_multiplying <= 1'b0;
      is_rounding <= 1'b0;
      is_finishing <= 1'b0;
    end else if (start) begin
      if (operation == FUNCTION_SCALE) begin
        is_multiplying <= 1'b1;
        step <= '0;
        multiplicand <= {32'b0, left[31] ? -left : left};
        multiplier <= right[31] ? -right : right;
        product <= '0;
        is_negative <= left[31] ^ right[31];
      end else begin
        exp2_index <= left[15:8];
        exp2_shift <= -($signed(left) >>> 16);
        is_exp2_saturated <= !left[31];
        rsqrt_index <= 8'(mantissa - 64);
        rsqrt_exponent <= even_exponent;
        is_rsqrt_zero <= left == 0;
        is_finishing <= 1'b1;
      end
    end else if (is_finishing) begin
      is_finishing <= 1'b0;
      result <= operation == FUNCTION_EXP2 ? exp2_result : rsqrt_result;
      done <= 1'b1;
    end else if (is_rounding) begin
      is_rounding <= 1'b0;
      result <= rounded_result;
      done <= 1'b1;
    end else if (is_multiplying) begin
      if (multiplier[0]) product <= product + multiplicand;
      multiplicand <= multiplicand << 1;
      multiplier <= multiplier >> 1;
      step <= step + 1;
      if (step == 6'd31) begin
        is_multiplying <= 1'b0;
        is_rounding <= 1'b1;
      end
    end
  end

endmodule

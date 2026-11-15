module decoder
  import tritium_package::*;
#(
    parameter logic [1:0] DOT_UNITS = DOT_TERNARY | DOT_INT8
) (
    input  logic     [31:0] word,
    output decoded_t        decoded
);

  logic [ 5:0] opcode;
  logic [31:0] signed_immediate;
  logic [31:0] unsigned_immediate;

  assign opcode = word[31:26];
  assign signed_immediate = {{16{word[15]}}, word[15:0]};
  assign unsigned_immediate = {16'b0, word[15:0]};

  always_comb begin
    decoded = '0;
    decoded.is_valid = 1'b1;
    decoded.immediate = signed_immediate;
    unique case (opcode)
      OPCODE_REGISTER: begin
        decoded.kind = KIND_REGISTER;
        decoded.operation = word[4:0];
        unique case (word[10:0])
          11'(FUNCTION_TERNARY_DOT): decoded.is_valid = DOT_UNITS[0];
          11'(FUNCTION_INT8_DOT): decoded.is_valid = DOT_UNITS[1];
          default: decoded.is_valid = word[10:0] <= 11'(FUNCTION_SATURATE_INT8);
        endcase
      end
      6'h01, 6'h08: begin
        decoded.kind = KIND_IMMEDIATE;
        decoded.operation = opcode == 6'h01 ? FUNCTION_ADD : FUNCTION_SET_LESS_THAN;
      end
      6'h02, 6'h03, 6'h04, 6'h05, 6'h06, 6'h07: begin
        decoded.kind = KIND_IMMEDIATE;
        decoded.operation = opcode[4:0];
        decoded.immediate = unsigned_immediate;
      end
      OPCODE_LOAD_UPPER: begin
        decoded.kind = KIND_LOAD_UPPER;
        decoded.immediate = {word[15:0], 16'b0};
      end
      OPCODE_READ_SPECIAL: begin
        decoded.kind = KIND_READ_SPECIAL;
        decoded.special = word[2:0];
        decoded.is_valid = word[15:0] <= 16'(SPECIAL_WARP_INDEX);
      end
      6'h10, 6'h11, 6'h12, 6'h13, 6'h14: begin
        decoded.kind = KIND_LOAD;
        decoded.width = 3'(opcode - 6'h10);
      end
      6'h15, 6'h16, 6'h17, 6'h18, 6'h19: begin
        decoded.kind = KIND_LOAD;
        decoded.width = 3'(opcode - 6'h15);
        decoded.is_shared = 1'b1;
      end
      6'h1A, 6'h1B, 6'h1C: begin
        decoded.kind = KIND_STORE;
        decoded.width = 3'(opcode - 6'h1A);
      end
      6'h1D, 6'h1E, 6'h1F: begin
        decoded.kind = KIND_STORE;
        decoded.width = 3'(opcode - 6'h1D);
        decoded.is_shared = 1'b1;
      end
      6'h20, 6'h21, 6'h22, 6'h23: begin
        decoded.kind = KIND_BRANCH;
        decoded.condition = opcode[1:0];
      end
      OPCODE_JUMP_AND_LINK: begin
        decoded.kind = KIND_JUMP_AND_LINK;
        decoded.immediate = {{11{word[20]}}, word[20:0]};
      end
      OPCODE_JUMP_AND_LINK_REGISTER: decoded.kind = KIND_JUMP_AND_LINK_REGISTER;
      OPCODE_BARRIER: decoded.kind = KIND_BARRIER;
      OPCODE_EXIT: decoded.kind = KIND_EXIT;
      default: decoded.is_valid = 1'b0;
    endcase
  end

endmodule

package tritium_package;

  localparam int LANES = 16;
  localparam int MAX_WARPS = 4;
  localparam int REGISTERS = 32;
  localparam int MAX_ARGUMENTS = 8;
  localparam int PROGRAM_WORDS = 4096;
  localparam int SHARED_WORDS = 4096;

  localparam int WARP_BITS = $clog2(MAX_WARPS);
  localparam int REGISTER_BITS = $clog2(REGISTERS);
  localparam int PROGRAM_BITS = $clog2(PROGRAM_WORDS);
  localparam int SHARED_BITS = $clog2(SHARED_WORDS);

  typedef enum logic [3:0] {
    FAULT_NONE = 4'd0,
    FAULT_INVALID_INSTRUCTION = 4'd1,
    FAULT_PROGRAM_COUNTER_OUT_OF_RANGE = 4'd2,
    FAULT_MISALIGNED_ACCESS = 4'd3,
    FAULT_ADDRESS_OUT_OF_RANGE = 4'd4,
    FAULT_DIVERGENT_BRANCH = 4'd5,
    FAULT_BARRIER_DEADLOCK = 4'd6,
    FAULT_STEP_LIMIT_EXCEEDED = 4'd7
  } fault_t;

  localparam logic [4:0] HOST_STATUS = 5'd0;
  localparam logic [4:0] HOST_GRID_BLOCKS = 5'd1;
  localparam logic [4:0] HOST_BLOCK_THREADS = 5'd2;
  localparam logic [4:0] HOST_SHARED_BYTES = 5'd3;
  localparam logic [4:0] HOST_ARGUMENT_COUNT = 5'd4;
  localparam logic [4:0] HOST_GLOBAL_BYTES = 5'd5;
  localparam logic [4:0] HOST_PROGRAM_LENGTH = 5'd6;
  localparam logic [4:0] HOST_STEP_LIMIT_LOW = 5'd7;
  localparam logic [4:0] HOST_STEP_LIMIT_HIGH = 5'd8;
  localparam logic [4:0] HOST_PROGRAM_ADDRESS = 5'd9;
  localparam logic [4:0] HOST_PROGRAM_DATA = 5'd10;
  localparam logic [4:0] HOST_ARGUMENTS = 5'd16;
  localparam logic [4:0] HOST_FAULT_BLOCK = 5'd24;
  localparam logic [4:0] HOST_FAULT_WARP = 5'd25;
  localparam logic [4:0] HOST_FAULT_PROGRAM_COUNTER = 5'd26;
  localparam logic [4:0] HOST_FAULT_VALUE = 5'd27;
  localparam logic [4:0] HOST_STEPS_LOW = 5'd28;
  localparam logic [4:0] HOST_STEPS_HIGH = 5'd29;

  localparam logic [5:0] OPCODE_REGISTER = 6'h00;
  localparam logic [5:0] OPCODE_LOAD_UPPER = 6'h09;
  localparam logic [5:0] OPCODE_READ_SPECIAL = 6'h0A;
  localparam logic [5:0] OPCODE_JUMP_AND_LINK_REGISTER = 6'h0B;
  localparam logic [5:0] OPCODE_JUMP_AND_LINK = 6'h0C;
  localparam logic [5:0] OPCODE_BARRIER = 6'h30;
  localparam logic [5:0] OPCODE_EXIT = 6'h31;

  typedef enum logic [3:0] {
    KIND_REGISTER,
    KIND_IMMEDIATE,
    KIND_LOAD,
    KIND_STORE,
    KIND_BRANCH,
    KIND_LOAD_UPPER,
    KIND_READ_SPECIAL,
    KIND_JUMP_AND_LINK,
    KIND_JUMP_AND_LINK_REGISTER,
    KIND_BARRIER,
    KIND_EXIT
  } kind_t;

  localparam logic [4:0] FUNCTION_ADD = 5'd0;
  localparam logic [4:0] FUNCTION_SUBTRACT = 5'd1;
  localparam logic [4:0] FUNCTION_AND = 5'd2;
  localparam logic [4:0] FUNCTION_OR = 5'd3;
  localparam logic [4:0] FUNCTION_XOR = 5'd4;
  localparam logic [4:0] FUNCTION_SHIFT_LEFT = 5'd5;
  localparam logic [4:0] FUNCTION_SHIFT_RIGHT_LOGICAL = 5'd6;
  localparam logic [4:0] FUNCTION_SHIFT_RIGHT_ARITHMETIC = 5'd7;
  localparam logic [4:0] FUNCTION_SET_LESS_THAN = 5'd8;
  localparam logic [4:0] FUNCTION_SET_LESS_THAN_UNSIGNED = 5'd9;
  localparam logic [4:0] FUNCTION_MINIMUM = 5'd10;
  localparam logic [4:0] FUNCTION_MAXIMUM = 5'd11;
  localparam logic [4:0] FUNCTION_MULTIPLY_HALFWORDS = 5'd12;
  localparam logic [4:0] FUNCTION_SELECT = 5'd13;
  localparam logic [4:0] FUNCTION_TERNARY_DOT = 5'd14;
  localparam logic [4:0] FUNCTION_SCALE = 5'd15;
  localparam logic [4:0] FUNCTION_EXP2 = 5'd16;
  localparam logic [4:0] FUNCTION_RECIPROCAL_SQUARE_ROOT = 5'd17;
  localparam logic [4:0] FUNCTION_SATURATE_INT8 = 5'd18;

  localparam logic [2:0] WIDTH_WORD = 3'd0;
  localparam logic [2:0] WIDTH_HALF = 3'd1;
  localparam logic [2:0] WIDTH_HALF_UNSIGNED = 3'd2;
  localparam logic [2:0] WIDTH_BYTE = 3'd3;
  localparam logic [2:0] WIDTH_BYTE_UNSIGNED = 3'd4;
  localparam logic [2:0] STORE_WIDTH_WORD = 3'd0;
  localparam logic [2:0] STORE_WIDTH_HALF = 3'd1;
  localparam logic [2:0] STORE_WIDTH_BYTE = 3'd2;

  localparam logic [2:0] SPECIAL_THREAD_INDEX = 3'd0;
  localparam logic [2:0] SPECIAL_BLOCK_INDEX = 3'd1;
  localparam logic [2:0] SPECIAL_BLOCK_THREADS = 3'd2;
  localparam logic [2:0] SPECIAL_GRID_BLOCKS = 3'd3;
  localparam logic [2:0] SPECIAL_LANE_INDEX = 3'd4;
  localparam logic [2:0] SPECIAL_WARP_INDEX = 3'd5;

  typedef struct packed {
    logic is_valid;
    kind_t kind;
    logic [4:0] operation;
    logic [31:0] immediate;
    logic [2:0] width;
    logic [2:0] special;
  } decoded_t;

endpackage

module compute_unit
  import tritium_package::*;
(
    input logic clock,
    input logic reset,

    input logic                    program_write,
    input logic [PROGRAM_BITS-1:0] program_address,
    input logic [            31:0] program_data,

    input logic        start,
    input logic [31:0] block_threads,
    input logic [31:0] program_length,
    input logic        is_step_allowed,

    output logic          busy,
    output logic          retire,
    output logic          fault,
    output fault_t        fault_kind,
    output logic   [31:0] fault_warp,
    output logic   [31:0] fault_program_counter,
    output logic   [31:0] fault_value
);

  typedef enum logic [2:0] {
    STATE_IDLE,
    STATE_SELECT,
    STATE_FETCH,
    STATE_DECODE,
    STATE_EXECUTE
  } state_t;

  typedef enum logic [1:0] {
    WARP_RUNNING,
    WARP_AT_BARRIER,
    WARP_EXITED
  } warp_status_t;

  state_t state;
  warp_status_t warp_status[MAX_WARPS];
  logic [31:0] program_counters[MAX_WARPS];
  logic [WARP_BITS-1:0] warp;
  logic [WARP_BITS:0] warp_count;

  logic [31:0] program_memory[PROGRAM_WORDS];
  logic [31:0] instruction;
  decoded_t decoded;

  logic [31:0] program_counter;
  logic is_any_running;
  logic [WARP_BITS-1:0] next_warp;

  assign program_counter = program_counters[warp];
  assign busy = state != STATE_IDLE;

  always_ff @(posedge clock) begin
    if (program_write) program_memory[program_address] <= program_data;
    instruction <= program_memory[program_counter[PROGRAM_BITS-1:0]];
  end

  decoder decoder (
      .opcode(instruction[31:26]),
      .decoded(decoded)
  );

  always_comb begin
    is_any_running = 1'b0;
    next_warp = warp;
    for (int offset = MAX_WARPS; offset >= 1; offset--) begin
      logic [WARP_BITS-1:0] candidate;
      candidate = warp + WARP_BITS'(offset);
      if ({1'b0, candidate} < warp_count && warp_status[candidate] == WARP_RUNNING) begin
        is_any_running = 1'b1;
        next_warp = candidate;
      end
    end
  end

  task automatic raise(input fault_t kind, input logic [31:0] value);
    fault <= 1'b1;
    fault_kind <= kind;
    fault_warp <= 32'(warp);
    fault_program_counter <= program_counter;
    fault_value <= value;
    state <= STATE_IDLE;
  endtask

  always_ff @(posedge clock) begin
    retire <= 1'b0;
    fault <= 1'b0;
    if (reset) begin
      state <= STATE_IDLE;
    end else begin
      unique case (state)
        STATE_IDLE: begin
          if (start) begin
            warp_count <= (WARP_BITS + 1)'(block_threads / LANES);
            for (int index = 0; index < MAX_WARPS; index++) begin
              warp_status[index] <= WARP_RUNNING;
              program_counters[index] <= '0;
            end
            warp <= WARP_BITS'(MAX_WARPS - 1);
            state <= STATE_SELECT;
          end
        end
        STATE_SELECT: begin
          if (is_any_running) begin
            warp <= next_warp;
            state <= STATE_FETCH;
          end else begin
            state <= STATE_IDLE;
          end
        end
        STATE_FETCH: begin
          if (!is_step_allowed) raise(FAULT_STEP_LIMIT_EXCEEDED, '0);
          else if (program_counter >= program_length)
            raise(FAULT_PROGRAM_COUNTER_OUT_OF_RANGE, '0);
          else state <= STATE_DECODE;
        end
        STATE_DECODE: begin
          if (!decoded.is_valid) raise(FAULT_INVALID_INSTRUCTION, instruction);
          else state <= STATE_EXECUTE;
        end
        STATE_EXECUTE: begin
          unique case (decoded.kind)
            KIND_EXIT: warp_status[warp] <= WARP_EXITED;
            default: ;
          endcase
          program_counters[warp] <= program_counter + 1;
          retire <= 1'b1;
          state <= STATE_SELECT;
        end
        default: state <= STATE_IDLE;
      endcase
    end
  end

endmodule

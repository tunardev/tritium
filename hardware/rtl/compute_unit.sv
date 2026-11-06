module compute_unit
  import tritium_package::*;
(
    input logic clock,
    input logic reset,

    input logic                    program_write,
    input logic [PROGRAM_BITS-1:0] program_address,
    input logic [            31:0] program_data,

    input logic                                start,
    input logic [                  31:0]       block_index,
    input logic [                  31:0]       grid_blocks,
    input logic [                  31:0]       block_threads,
    input logic [                  31:0]       global_bytes,
    input logic [                  31:0]       shared_bytes,
    input logic [                  31:0]       program_length,
    input logic [                   3:0]       argument_count,
    input logic [MAX_ARGUMENTS-1:0][31:0]       arguments,
    input logic                                is_step_allowed,

    output logic          busy,
    output logic          retire,
    output logic          fault,
    output fault_t        fault_kind,
    output logic   [31:0] fault_warp,
    output logic   [31:0] fault_program_counter,
    output logic   [31:0] fault_value,

    output logic        memory_request,
    output logic        memory_write,
    output logic [31:0] memory_address,
    output logic [ 3:0] memory_byte_enable,
    output logic [31:0] memory_write_data,
    input  logic        memory_response,
    input  logic [31:0] memory_read_data
);

  localparam int REGISTER_ADDRESS_BITS = WARP_BITS + REGISTER_BITS;

  typedef enum logic [3:0] {
    STATE_IDLE,
    STATE_INITIALIZE,
    STATE_SELECT,
    STATE_FETCH,
    STATE_DECODE,
    STATE_EXECUTE,
    STATE_MEMORY_ADDRESS,
    STATE_MEMORY_WAIT,
    STATE_DOT_ADDRESS,
    STATE_DOT_WEIGHTS,
    STATE_DOT_READ,
    STATE_DOT_CAPTURE,
    STATE_DOT_WRITE,
    STATE_FIXED_START,
    STATE_FIXED_WAIT
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
  logic [SHARED_BITS:0] initialize_index;
  logic [SHARED_BITS:0] initialize_count;
  logic [REGISTER_BITS-1:0] initialize_register;
  logic [$clog2(LANES)-1:0] lane;

  logic [31:0] program_memory[PROGRAM_WORDS];
  logic [31:0] instruction;
  decoded_t decoded;
  decoded_t current;
  logic [4:0] destination;

  logic [31:0] program_counter;
  logic [31:0] link_address;
  logic is_any_running;
  logic is_any_at_barrier;
  logic is_any_exited;
  logic [WARP_BITS-1:0] next_warp;
  logic [LANES-1:0] is_taken;
  logic [LANES-1:0][31:0] jump_targets;
  logic is_branch_uniform;
  logic is_jump_uniform;

  logic [LANES-1:0][31:0] read_data_a;
  logic [LANES-1:0][31:0] read_data_b;
  logic [LANES-1:0][31:0] read_data_c;
  logic [LANES-1:0] write_enable;
  logic [REGISTER_ADDRESS_BITS-1:0] write_address;
  logic [LANES-1:0][31:0] write_data;
  logic [LANES-1:0][31:0] lane_results;

  logic [31:0] lane_address;
  logic [31:0] space_bytes;
  logic [2:0] access_bytes;
  logic [3:0] byte_enable;
  logic is_access_complete;
  logic [31:0] access_read_data;
  logic [SHARED_BITS-1:0] shared_address;
  logic [3:0] shared_write_enable;
  logic [31:0] shared_write_data;
  logic [31:0] shared_read_data;

  logic [31:0] dot_weights;
  logic [5:0][31:0] dot_words;
  logic [2:0] dot_word;
  logic [SHARED_BITS-1:0] dot_word_address;
  logic [19:0][7:0] dot_activations;
  logic [31:0] dot_sum;
  logic [31:0] dot_result;
  logic is_serial;
  logic fixed_done;
  logic [31:0] fixed_result;
  logic [31:0] shifted_read_data;
  logic [31:0] loaded_value;

  assign program_counter = program_counters[warp];
  assign link_address = program_counter + 1;
  assign initialize_register = initialize_index[REGISTER_BITS-1:0];
  assign is_branch_uniform = is_taken == '0 || is_taken == '1;
  assign busy = state != STATE_IDLE;
  assign destination = instruction[25:21];

  always_ff @(posedge clock) begin
    if (program_write) program_memory[program_address] <= program_data;
    instruction <= program_memory[program_counter[PROGRAM_BITS-1:0]];
  end

  decoder decoder (
      .word(instruction),
      .decoded(decoded)
  );

  register_file register_file (
      .clock(clock),
      .read(state == STATE_DECODE),
      .read_address_a({warp, instruction[25:21]}),
      .read_address_b({warp, instruction[20:16]}),
      .read_address_c({warp, instruction[15:11]}),
      .read_data_a(read_data_a),
      .read_data_b(read_data_b),
      .read_data_c(read_data_c),
      .write_enable(write_enable),
      .write_address(write_address),
      .write_data(write_data)
  );

  shared_memory shared_memory (
      .clock(clock),
      .address(shared_address),
      .write_enable(shared_write_enable),
      .write_data(shared_write_data),
      .read_data(shared_read_data)
  );

  assign dot_word_address = read_data_c[lane][SHARED_BITS+1:2] + SHARED_BITS'(dot_word);
  assign dot_activations = 160'(dot_words >> {read_data_c[lane][1:0], 3'b0});
  assign dot_result = read_data_a[lane] + dot_sum;
  assign is_serial = current.kind == KIND_REGISTER && current.operation inside {
      FUNCTION_TERNARY_DOT, FUNCTION_SCALE, FUNCTION_EXP2, FUNCTION_RECIPROCAL_SQUARE_ROOT};

  fixed_point_unit fixed_point_unit (
      .clock(clock),
      .reset(reset),
      .start(state == STATE_FIXED_START),
      .operation(current.operation),
      .left(read_data_b[lane]),
      .right(read_data_c[lane]),
      .done(fixed_done),
      .result(fixed_result)
  );

  ternary_dot ternary_dot (
      .weights(dot_weights),
      .activations(dot_activations),
      .sum(dot_sum)
  );

  for (genvar index = 0; index < LANES; index++) begin : gen_lanes
    logic [31:0] alu_result;
    logic [31:0] special;

    lane_alu lane_alu (
        .operation(current.operation),
        .left(read_data_b[index]),
        .right(current.kind == KIND_IMMEDIATE ? current.immediate : read_data_c[index]),
        .destination(read_data_a[index]),
        .result(alu_result)
    );

    always_comb begin
      unique case (current.condition)
        CONDITION_EQUAL: is_taken[index] = read_data_a[index] == read_data_b[index];
        CONDITION_NOT_EQUAL: is_taken[index] = read_data_a[index] != read_data_b[index];
        CONDITION_LESS_THAN:
        is_taken[index] = $signed(read_data_a[index]) < $signed(read_data_b[index]);
        default: is_taken[index] = $signed(read_data_a[index]) >= $signed(read_data_b[index]);
      endcase
      jump_targets[index] = read_data_b[index] + current.immediate;
      unique case (current.special)
        SPECIAL_THREAD_INDEX: special = 32'(warp) * LANES + index;
        SPECIAL_BLOCK_INDEX: special = block_index;
        SPECIAL_BLOCK_THREADS: special = block_threads;
        SPECIAL_GRID_BLOCKS: special = grid_blocks;
        SPECIAL_LANE_INDEX: special = index;
        default: special = 32'(warp);
      endcase
      unique case (current.kind)
        KIND_LOAD_UPPER: lane_results[index] = current.immediate;
        KIND_READ_SPECIAL: lane_results[index] = special;
        KIND_JUMP_AND_LINK, KIND_JUMP_AND_LINK_REGISTER: lane_results[index] = link_address;
        default: lane_results[index] = alu_result;
      endcase
    end
  end

  always_comb begin
    is_jump_uniform = 1'b1;
    for (int index = 1; index < LANES; index++)
      if (jump_targets[index] != jump_targets[0]) is_jump_uniform = 1'b0;
  end

  always_comb begin
    is_any_running = 1'b0;
    is_any_at_barrier = 1'b0;
    is_any_exited = 1'b0;
    next_warp = warp;
    for (int offset = MAX_WARPS; offset >= 1; offset--) begin
      logic [WARP_BITS-1:0] candidate;
      candidate = warp + WARP_BITS'(offset);
      if ({1'b0, candidate} < warp_count) begin
        if (warp_status[candidate] == WARP_RUNNING) begin
          is_any_running = 1'b1;
          next_warp = candidate;
        end
        if (warp_status[candidate] == WARP_AT_BARRIER) is_any_at_barrier = 1'b1;
        if (warp_status[candidate] == WARP_EXITED) is_any_exited = 1'b1;
      end
    end
  end

  always_comb begin
    lane_address = read_data_b[lane] + current.immediate;
    if (current.kind == KIND_STORE) begin
      unique case (current.width)
        STORE_WIDTH_WORD: access_bytes = 3'd4;
        STORE_WIDTH_HALF: access_bytes = 3'd2;
        default: access_bytes = 3'd1;
      endcase
    end else begin
      unique case (current.width)
        WIDTH_WORD: access_bytes = 3'd4;
        WIDTH_HALF, WIDTH_HALF_UNSIGNED: access_bytes = 3'd2;
        default: access_bytes = 3'd1;
      endcase
    end
    unique case (access_bytes)
      3'd4: byte_enable = 4'b1111;
      3'd2: byte_enable = 4'b0011 << lane_address[1:0];
      default: byte_enable = 4'b0001 << lane_address[1:0];
    endcase
    space_bytes = current.is_shared ? shared_bytes : global_bytes;
    is_access_complete = current.is_shared || memory_response;
    access_read_data = current.is_shared ? shared_read_data : memory_read_data;
    shifted_read_data = access_read_data >> {lane_address[1:0], 3'b0};
    unique case (current.width)
      WIDTH_HALF: loaded_value = {{16{shifted_read_data[15]}}, shifted_read_data[15:0]};
      WIDTH_HALF_UNSIGNED: loaded_value = {16'b0, shifted_read_data[15:0]};
      WIDTH_BYTE: loaded_value = {{24{shifted_read_data[7]}}, shifted_read_data[7:0]};
      WIDTH_BYTE_UNSIGNED: loaded_value = {24'b0, shifted_read_data[7:0]};
      default: loaded_value = shifted_read_data;
    endcase
  end

  always_comb begin
    shared_address = lane_address[SHARED_BITS+1:2];
    shared_write_enable = '0;
    shared_write_data = read_data_a[lane] << {lane_address[1:0], 3'b0};
    if (state == STATE_INITIALIZE) begin
      shared_address = initialize_index[SHARED_BITS-1:0];
      shared_write_enable = '1;
      shared_write_data = '0;
    end else if (state == STATE_MEMORY_ADDRESS && current.is_shared && current.kind == KIND_STORE)
      shared_write_enable = byte_enable;
    else if (state == STATE_DOT_READ) shared_address = dot_word_address;
  end

  always_comb begin
    write_enable = '0;
    write_address = {warp, destination};
    write_data = lane_results;
    unique case (state)
      STATE_INITIALIZE: begin
        write_enable = {LANES{initialize_index < 2 ** REGISTER_ADDRESS_BITS}};
        write_address = initialize_index[REGISTER_ADDRESS_BITS-1:0];
        for (int index = 0; index < LANES; index++) begin
          write_data[index] = '0;
          if (initialize_register != 0 && 4'(initialize_register) <= argument_count)
            write_data[index] = arguments[initialize_register-1];
        end
      end
      STATE_EXECUTE: begin
        if (destination != 0 && !is_serial && current.kind inside {KIND_REGISTER, KIND_IMMEDIATE,
            KIND_LOAD_UPPER, KIND_READ_SPECIAL, KIND_JUMP_AND_LINK})
          write_enable = '1;
        if (destination != 0 && current.kind == KIND_JUMP_AND_LINK_REGISTER && is_jump_uniform)
          write_enable = '1;
      end
      STATE_MEMORY_WAIT: begin
        if (is_access_complete && current.kind == KIND_LOAD && destination != 0)
          write_enable[lane] = 1'b1;
        write_data = {LANES{loaded_value}};
      end
      STATE_DOT_WRITE: begin
        write_enable[lane] = destination != 0;
        write_data = {LANES{dot_result}};
      end
      STATE_FIXED_WAIT: begin
        write_enable[lane] = fixed_done && destination != 0;
        write_data = {LANES{fixed_result}};
      end
      default: ;
    endcase
  end

  task automatic raise(input fault_t kind, input logic [31:0] value);
    fault <= 1'b1;
    fault_kind <= kind;
    fault_warp <= 32'(warp);
    fault_program_counter <= program_counter;
    fault_value <= value;
    state <= STATE_IDLE;
  endtask

  task automatic finish_instruction(input logic [31:0] next_program_counter);
    program_counters[warp] <= next_program_counter;
    retire <= 1'b1;
    state <= STATE_SELECT;
  endtask

  always_ff @(posedge clock) begin
    retire <= 1'b0;
    fault <= 1'b0;
    memory_request <= 1'b0;
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
            initialize_index <= '0;
            initialize_count <= (SHARED_BITS + 1)'((shared_bytes + 3) / 4);
            if ((shared_bytes + 3) / 4 < 2 ** REGISTER_ADDRESS_BITS)
              initialize_count <= 2 ** REGISTER_ADDRESS_BITS;
            state <= STATE_INITIALIZE;
          end
        end
        STATE_INITIALIZE: begin
          initialize_index <= initialize_index + 1;
          if (initialize_index == initialize_count - 1) state <= STATE_SELECT;
        end
        STATE_SELECT: begin
          if (is_any_running) begin
            warp <= next_warp;
            state <= STATE_FETCH;
          end else if (is_any_at_barrier && is_any_exited) begin
            raise(FAULT_BARRIER_DEADLOCK, '0);
          end else if (is_any_at_barrier) begin
            for (int index = 0; index < MAX_WARPS; index++)
              if (warp_status[index] == WARP_AT_BARRIER) warp_status[index] <= WARP_RUNNING;
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
          current <= decoded;
          if (!decoded.is_valid) raise(FAULT_INVALID_INSTRUCTION, instruction);
          else state <= STATE_EXECUTE;
        end
        STATE_EXECUTE: begin
          unique case (current.kind)
            KIND_LOAD, KIND_STORE: begin
              lane <= '0;
              state <= STATE_MEMORY_ADDRESS;
            end
            KIND_REGISTER: begin
              lane <= '0;
              if (!is_serial) finish_instruction(link_address);
              else if (current.operation == FUNCTION_TERNARY_DOT) state <= STATE_DOT_ADDRESS;
              else state <= STATE_FIXED_START;
            end
            KIND_BRANCH: begin
              if (!is_branch_uniform) raise(FAULT_DIVERGENT_BRANCH, '0);
              else if (is_taken[0]) finish_instruction(program_counter + current.immediate);
              else finish_instruction(link_address);
            end
            KIND_JUMP_AND_LINK: finish_instruction(program_counter + current.immediate);
            KIND_JUMP_AND_LINK_REGISTER: begin
              if (!is_jump_uniform) raise(FAULT_DIVERGENT_BRANCH, '0);
              else finish_instruction(jump_targets[0]);
            end
            KIND_BARRIER: begin
              warp_status[warp] <= WARP_AT_BARRIER;
              finish_instruction(link_address);
            end
            KIND_EXIT: begin
              warp_status[warp] <= WARP_EXITED;
              finish_instruction(link_address);
            end
            default: finish_instruction(link_address);
          endcase
        end
        STATE_MEMORY_ADDRESS: begin
          if ((lane_address & 32'(access_bytes - 1)) != 0)
            raise(FAULT_MISALIGNED_ACCESS, lane_address);
          else if ({1'b0, lane_address} + 33'(access_bytes) > {1'b0, space_bytes})
            raise(FAULT_ADDRESS_OUT_OF_RANGE, lane_address);
          else if (current.is_shared) begin
            state <= STATE_MEMORY_WAIT;
          end else begin
            memory_request <= 1'b1;
            memory_write <= current.kind == KIND_STORE;
            memory_address <= {lane_address[31:2], 2'b0};
            memory_write_data <= read_data_a[lane] << {lane_address[1:0], 3'b0};
            memory_byte_enable <= byte_enable;
            state <= STATE_MEMORY_WAIT;
          end
        end
        STATE_MEMORY_WAIT: begin
          if (is_access_complete) begin
            lane <= lane + 1;
            if (lane == '1) finish_instruction(link_address);
            else state <= STATE_MEMORY_ADDRESS;
          end
        end
        STATE_DOT_ADDRESS: begin
          if (read_data_b[lane][1:0] != 0) raise(FAULT_MISALIGNED_ACCESS, read_data_b[lane]);
          else if ({1'b0, read_data_b[lane]} + 33'd4 > {1'b0, global_bytes})
            raise(FAULT_ADDRESS_OUT_OF_RANGE, read_data_b[lane]);
          else if ({1'b0, read_data_c[lane]} + 33'd20 > {1'b0, shared_bytes})
            raise(FAULT_ADDRESS_OUT_OF_RANGE, read_data_c[lane]);
          else begin
            memory_request <= 1'b1;
            memory_write <= 1'b0;
            memory_address <= read_data_b[lane];
            memory_byte_enable <= 4'b0000;
            state <= STATE_DOT_WEIGHTS;
          end
        end
        STATE_DOT_WEIGHTS: begin
          if (memory_response) begin
            dot_weights <= memory_read_data;
            dot_word <= '0;
            state <= STATE_DOT_READ;
          end
        end
        STATE_DOT_READ: state <= STATE_DOT_CAPTURE;
        STATE_DOT_CAPTURE: begin
          dot_words[dot_word] <= shared_read_data;
          dot_word <= dot_word + 1;
          state <= dot_word == 3'd5 ? STATE_DOT_WRITE : STATE_DOT_READ;
        end
        STATE_DOT_WRITE: begin
          lane <= lane + 1;
          if (lane == '1) finish_instruction(link_address);
          else state <= STATE_DOT_ADDRESS;
        end
        STATE_FIXED_START: state <= STATE_FIXED_WAIT;
        STATE_FIXED_WAIT: begin
          if (fixed_done) begin
            lane <= lane + 1;
            if (lane == '1) finish_instruction(link_address);
            else state <= STATE_FIXED_START;
          end
        end
        default: state <= STATE_IDLE;
      endcase
    end
  end

endmodule

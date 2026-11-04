module tritium_gpu
  import tritium_package::*;
(
    input logic clock,
    input logic reset,

    input  logic        host_write,
    input  logic [ 4:0] host_address,
    input  logic [31:0] host_write_data,
    output logic [31:0] host_read_data,

    output logic        memory_request,
    output logic        memory_write,
    output logic [31:0] memory_address,
    output logic [ 3:0] memory_byte_enable,
    output logic [31:0] memory_write_data,
    input  logic        memory_response,
    input  logic [31:0] memory_read_data
);

  logic [31:0] grid_blocks;
  logic [31:0] block_threads;
  logic [31:0] global_bytes;
  logic [31:0] shared_bytes;
  logic [3:0] argument_count;
  logic [MAX_ARGUMENTS-1:0][31:0] arguments;
  logic [31:0] program_length;
  logic [63:0] step_limit;
  logic [63:0] steps;
  logic [PROGRAM_BITS-1:0] program_address;

  logic is_busy;
  logic is_done;
  logic [31:0] next_block;
  logic [31:0] unit_block;
  logic unit_start;

  fault_t fault_kind;
  logic [31:0] fault_block;
  logic [31:0] fault_warp;
  logic [31:0] fault_program_counter;
  logic [31:0] fault_value;

  logic unit_busy;
  logic unit_retire;
  logic unit_fault;
  fault_t unit_fault_kind;
  logic [31:0] unit_fault_warp;
  logic [31:0] unit_fault_program_counter;
  logic [31:0] unit_fault_value;

  logic program_write;

  assign program_write = host_write && host_address == HOST_PROGRAM_DATA;

  compute_unit compute_unit (
      .clock(clock),
      .reset(reset),
      .program_write(program_write),
      .program_address(program_address),
      .program_data(host_write_data),
      .start(unit_start),
      .block_index(unit_block),
      .grid_blocks(grid_blocks),
      .block_threads(block_threads),
      .global_bytes(global_bytes),
      .shared_bytes(shared_bytes),
      .program_length(program_length),
      .argument_count(argument_count),
      .arguments(arguments),
      .is_step_allowed(steps < step_limit),
      .busy(unit_busy),
      .retire(unit_retire),
      .fault(unit_fault),
      .fault_kind(unit_fault_kind),
      .fault_warp(unit_fault_warp),
      .fault_program_counter(unit_fault_program_counter),
      .fault_value(unit_fault_value),
      .memory_request(memory_request),
      .memory_write(memory_write),
      .memory_address(memory_address),
      .memory_byte_enable(memory_byte_enable),
      .memory_write_data(memory_write_data),
      .memory_response(memory_response),
      .memory_read_data(memory_read_data)
  );

  always_ff @(posedge clock) begin
    unit_start <= 1'b0;
    if (reset) begin
      is_busy <= 1'b0;
      is_done <= 1'b0;
      fault_kind <= FAULT_NONE;
      program_address <= '0;
    end else begin
      if (host_write) begin
        unique case (host_address)
          HOST_STATUS: begin
            if (host_write_data[0]) begin
              is_busy <= 1'b1;
              is_done <= 1'b0;
              fault_kind <= FAULT_NONE;
              next_block <= '0;
              steps <= '0;
            end
          end
          HOST_GRID_BLOCKS: grid_blocks <= host_write_data;
          HOST_BLOCK_THREADS: block_threads <= host_write_data;
          HOST_ARGUMENT_COUNT: argument_count <= host_write_data[3:0];
          HOST_GLOBAL_BYTES: global_bytes <= host_write_data;
          HOST_SHARED_BYTES: shared_bytes <= host_write_data;
          HOST_PROGRAM_LENGTH: program_length <= host_write_data;
          HOST_STEP_LIMIT_LOW: step_limit[31:0] <= host_write_data;
          HOST_STEP_LIMIT_HIGH: step_limit[63:32] <= host_write_data;
          HOST_PROGRAM_ADDRESS: program_address <= host_write_data[PROGRAM_BITS-1:0];
          HOST_PROGRAM_DATA: program_address <= program_address + 1;
          default: begin
            if (host_address[4:3] == HOST_ARGUMENTS[4:3])
              arguments[host_address[2:0]] <= host_write_data;
          end
        endcase
      end

      if (unit_retire) steps <= steps + 1;

      if (unit_fault) begin
        fault_kind <= unit_fault_kind;
        fault_block <= unit_block;
        fault_warp <= unit_fault_warp;
        fault_program_counter <= unit_fault_program_counter;
        fault_value <= unit_fault_value;
      end

      if (is_busy && !unit_busy && !unit_start && !unit_fault) begin
        if (next_block < grid_blocks && fault_kind == FAULT_NONE) begin
          unit_block <= next_block;
          unit_start <= 1'b1;
          next_block <= next_block + 1;
        end else begin
          is_busy <= 1'b0;
          is_done <= 1'b1;
        end
      end
    end
  end

  always_comb begin
    unique case (host_address)
      HOST_STATUS: host_read_data = {24'b0, fault_kind, 2'b0, is_done, is_busy};
      HOST_FAULT_BLOCK: host_read_data = fault_block;
      HOST_FAULT_WARP: host_read_data = fault_warp;
      HOST_FAULT_PROGRAM_COUNTER: host_read_data = fault_program_counter;
      HOST_FAULT_VALUE: host_read_data = fault_value;
      HOST_STEPS_LOW: host_read_data = steps[31:0];
      HOST_STEPS_HIGH: host_read_data = steps[63:32];
      default: host_read_data = '0;
    endcase
  end

endmodule

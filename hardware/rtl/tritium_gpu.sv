module tritium_gpu
  import tritium_package::*;
#(
    parameter int COMPUTE_UNITS = 4
) (
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
    input  logic        memory_ready,
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

  fault_t fault_kind;
  logic [31:0] fault_block;
  logic [31:0] fault_warp;
  logic [31:0] fault_program_counter;
  logic [31:0] fault_value;

  logic [COMPUTE_UNITS-1:0] unit_start;
  logic [COMPUTE_UNITS-1:0][31:0] unit_block;
  logic [COMPUTE_UNITS-1:0] unit_busy;
  logic [COMPUTE_UNITS-1:0] unit_retire;
  logic [COMPUTE_UNITS-1:0] unit_fault;
  fault_t [COMPUTE_UNITS-1:0] unit_fault_kind;
  logic [COMPUTE_UNITS-1:0][31:0] unit_fault_warp;
  logic [COMPUTE_UNITS-1:0][31:0] unit_fault_program_counter;
  logic [COMPUTE_UNITS-1:0][31:0] unit_fault_value;
  logic [COMPUTE_UNITS-1:0] unit_memory_request;
  logic [COMPUTE_UNITS-1:0] unit_memory_write;
  logic [COMPUTE_UNITS-1:0][31:0] unit_memory_address;
  logic [COMPUTE_UNITS-1:0][3:0] unit_memory_byte_enable;
  logic [COMPUTE_UNITS-1:0][31:0] unit_memory_write_data;
  logic [COMPUTE_UNITS-1:0] unit_memory_response;

  logic program_write;
  logic is_step_allowed;
  logic [$clog2(COMPUTE_UNITS+1)-1:0] retired;
  logic is_any_idle;
  logic [$clog2(COMPUTE_UNITS)-1:0] idle_unit;
  logic is_any_fault;
  logic [$clog2(COMPUTE_UNITS)-1:0] fault_unit;

  assign program_write = host_write && host_address == HOST_PROGRAM_DATA;
  assign is_step_allowed = steps < step_limit;

  for (genvar unit = 0; unit < COMPUTE_UNITS; unit++) begin : gen_units
    compute_unit compute_unit (
        .clock(clock),
        .reset(reset),
        .program_write(program_write),
        .program_address(program_address),
        .program_data(host_write_data),
        .start(unit_start[unit]),
        .block_index(unit_block[unit]),
        .grid_blocks(grid_blocks),
        .block_threads(block_threads),
        .global_bytes(global_bytes),
        .shared_bytes(shared_bytes),
        .program_length(program_length),
        .argument_count(argument_count),
        .arguments(arguments),
        .is_step_allowed(is_step_allowed),
        .busy(unit_busy[unit]),
        .retire(unit_retire[unit]),
        .fault(unit_fault[unit]),
        .fault_kind(unit_fault_kind[unit]),
        .fault_warp(unit_fault_warp[unit]),
        .fault_program_counter(unit_fault_program_counter[unit]),
        .fault_value(unit_fault_value[unit]),
        .memory_request(unit_memory_request[unit]),
        .memory_write(unit_memory_write[unit]),
        .memory_address(unit_memory_address[unit]),
        .memory_byte_enable(unit_memory_byte_enable[unit]),
        .memory_write_data(unit_memory_write_data[unit]),
        .memory_response(unit_memory_response[unit]),
        .memory_read_data(memory_read_data)
    );
  end

  memory_arbiter #(
      .UNITS(COMPUTE_UNITS)
  ) memory_arbiter (
      .clock(clock),
      .reset(reset),
      .unit_request(unit_memory_request),
      .unit_write(unit_memory_write),
      .unit_address(unit_memory_address),
      .unit_byte_enable(unit_memory_byte_enable),
      .unit_write_data(unit_memory_write_data),
      .unit_response(unit_memory_response),
      .memory_request(memory_request),
      .memory_write(memory_write),
      .memory_address(memory_address),
      .memory_byte_enable(memory_byte_enable),
      .memory_write_data(memory_write_data),
      .memory_ready(memory_ready),
      .memory_response(memory_response)
  );

  always_comb begin
    retired = '0;
    is_any_idle = 1'b0;
    idle_unit = '0;
    is_any_fault = 1'b0;
    fault_unit = '0;
    for (int unit = COMPUTE_UNITS - 1; unit >= 0; unit--) begin
      retired = retired + unit_retire[unit];
      if (!unit_busy[unit] && !unit_start[unit]) begin
        is_any_idle = 1'b1;
        idle_unit = unit[$clog2(COMPUTE_UNITS)-1:0];
      end
      if (unit_fault[unit]) begin
        is_any_fault = 1'b1;
        fault_unit = unit[$clog2(COMPUTE_UNITS)-1:0];
      end
    end
  end

  always_ff @(posedge clock) begin
    unit_start <= '0;
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

      steps <= steps + 64'(retired);

      if (is_any_fault && fault_kind == FAULT_NONE) begin
        fault_kind <= unit_fault_kind[fault_unit];
        fault_block <= unit_block[fault_unit];
        fault_warp <= unit_fault_warp[fault_unit];
        fault_program_counter <= unit_fault_program_counter[fault_unit];
        fault_value <= unit_fault_value[fault_unit];
      end

      if (is_busy) begin
        if (next_block < grid_blocks && fault_kind == FAULT_NONE && !is_any_fault) begin
          if (is_any_idle) begin
            unit_block[idle_unit] <= next_block;
            unit_start[idle_unit] <= 1'b1;
            next_block <= next_block + 1;
          end
        end else if (unit_busy == '0 && unit_start == '0 && unit_fault == '0) begin
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

module memory_arbiter #(
    parameter int UNITS = 2,
    parameter int OUTSTANDING = 32
) (
    input logic clock,
    input logic reset,

    input  logic [UNITS-1:0]       unit_request,
    input  logic [UNITS-1:0]       unit_write,
    input  logic [UNITS-1:0][31:0] unit_address,
    input  logic [UNITS-1:0][ 3:0] unit_byte_enable,
    input  logic [UNITS-1:0][31:0] unit_write_data,
    output logic [UNITS-1:0]       unit_accepted,
    output logic [UNITS-1:0]       unit_response,

    output logic        memory_request,
    output logic        memory_write,
    output logic [31:0] memory_address,
    output logic [ 3:0] memory_byte_enable,
    output logic [31:0] memory_write_data,
    input  logic        memory_ready,
    input  logic        memory_response
);

  localparam int UNIT_BITS = UNITS > 1 ? $clog2(UNITS) : 1;
  localparam int QUEUE_BITS = $clog2(OUTSTANDING);

  logic [UNIT_BITS-1:0] owners[OUTSTANDING];
  logic [QUEUE_BITS-1:0] owner_head;
  logic [QUEUE_BITS-1:0] owner_tail;
  logic [UNIT_BITS-1:0] last_grant;

  logic is_granting;
  logic [UNIT_BITS-1:0] grant;

  always_comb begin
    is_granting = 1'b0;
    grant = last_grant;
    for (int offset = UNITS; offset >= 1; offset--) begin
      logic [UNIT_BITS-1:0] candidate;
      candidate = UNIT_BITS'((32'(last_grant) + offset) % UNITS);
      if (unit_request[candidate]) begin
        is_granting = memory_ready;
        grant = candidate;
      end
    end
    unit_accepted = '0;
    if (is_granting) unit_accepted[grant] = 1'b1;
    unit_response = '0;
    if (memory_response) unit_response[owners[owner_head]] = 1'b1;
  end

  always_ff @(posedge clock) begin
    memory_request <= 1'b0;
    if (reset) begin
      owner_head <= '0;
      owner_tail <= '0;
      last_grant <= '0;
    end else begin
      if (is_granting) begin
        memory_request <= 1'b1;
        memory_write <= unit_write[grant];
        memory_address <= unit_address[grant];
        memory_byte_enable <= unit_byte_enable[grant];
        memory_write_data <= unit_write_data[grant];
        owners[owner_tail] <= grant;
        owner_tail <= owner_tail + 1;
        last_grant <= grant;
      end
      if (memory_response) owner_head <= owner_head + 1;
    end
  end

endmodule

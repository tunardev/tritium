module memory_arbiter #(
    parameter int UNITS = 4
) (
    input logic clock,
    input logic reset,

    input  logic [UNITS-1:0]       unit_request,
    input  logic [UNITS-1:0]       unit_write,
    input  logic [UNITS-1:0][31:0] unit_address,
    input  logic [UNITS-1:0][ 3:0] unit_byte_enable,
    input  logic [UNITS-1:0][31:0] unit_write_data,
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

  logic [UNITS-1:0] is_pending;
  logic [UNITS-1:0] pending_write;
  logic [UNITS-1:0][31:0] pending_address;
  logic [UNITS-1:0][3:0] pending_byte_enable;
  logic [UNITS-1:0][31:0] pending_write_data;

  logic [UNIT_BITS-1:0] owners[UNITS];
  logic [UNIT_BITS-1:0] owner_head;
  logic [UNIT_BITS-1:0] owner_tail;
  logic [UNIT_BITS-1:0] last_grant;

  logic is_granting;
  logic [UNIT_BITS-1:0] grant;

  always_comb begin
    is_granting = 1'b0;
    grant = last_grant;
    for (int offset = UNITS; offset >= 1; offset--) begin
      logic [UNIT_BITS-1:0] candidate;
      candidate = UNIT_BITS'((32'(last_grant) + offset) % UNITS);
      if (is_pending[candidate]) begin
        is_granting = memory_ready;
        grant = candidate;
      end
    end
    unit_response = '0;
    if (memory_response) unit_response[owners[owner_head]] = 1'b1;
  end

  always_ff @(posedge clock) begin
    memory_request <= 1'b0;
    if (reset) begin
      is_pending <= '0;
      owner_head <= '0;
      owner_tail <= '0;
      last_grant <= '0;
    end else begin
      for (int unit = 0; unit < UNITS; unit++) begin
        if (unit_request[unit]) begin
          is_pending[unit] <= 1'b1;
          pending_write[unit] <= unit_write[unit];
          pending_address[unit] <= unit_address[unit];
          pending_byte_enable[unit] <= unit_byte_enable[unit];
          pending_write_data[unit] <= unit_write_data[unit];
        end
      end
      if (is_granting) begin
        is_pending[grant] <= 1'b0;
        memory_request <= 1'b1;
        memory_write <= pending_write[grant];
        memory_address <= pending_address[grant];
        memory_byte_enable <= pending_byte_enable[grant];
        memory_write_data <= pending_write_data[grant];
        owners[owner_tail] <= grant;
        owner_tail <= UNIT_BITS'((32'(owner_tail) + 1) % UNITS);
        last_grant <= grant;
      end
      if (memory_response) owner_head <= UNIT_BITS'((32'(owner_head) + 1) % UNITS);
    end
  end

endmodule

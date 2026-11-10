module register_file
  import tritium_package::*;
#(
    localparam int ADDRESS_BITS = WARP_BITS + REGISTER_BITS
) (
    input logic clock,

    input  logic                               read_first,
    input  logic                               read_second,
    input  logic [   ADDRESS_BITS-1:0]         read_address_a,
    input  logic [   ADDRESS_BITS-1:0]         read_address_b,
    input  logic [   ADDRESS_BITS-1:0]         read_address_c,
    output logic [          LANES-1:0][31:0] read_data_a,
    output logic [          LANES-1:0][31:0] read_data_b,
    output logic [          LANES-1:0][31:0] read_data_c,

    input logic [       LANES-1:0]       write_enable,
    input logic [ADDRESS_BITS-1:0]       write_address,
    input logic [       LANES-1:0][31:0] write_data
);

  logic [ADDRESS_BITS-1:0] first_address;
  logic [ADDRESS_BITS-1:0] second_address;

  assign first_address = write_enable != '0 ? write_address : read_address_a;
  assign second_address = read_second ? read_address_c : read_address_b;

  for (genvar lane = 0; lane < LANES; lane++) begin : gen_lanes
    logic [31:0] registers[2**ADDRESS_BITS];
    logic [31:0] second_data;

    always_ff @(posedge clock) begin
      if (write_enable[lane]) registers[first_address] <= write_data[lane];
      if (read_first) read_data_a[lane] <= registers[first_address];
      if (read_first || read_second) second_data <= registers[second_address];
      if (read_second) read_data_b[lane] <= second_data;
    end

    assign read_data_c[lane] = second_data;
  end

endmodule

module register_file
  import tritium_package::*;
#(
    localparam int ADDRESS_BITS = WARP_BITS + REGISTER_BITS
) (
    input logic clock,

    input  logic                               read,
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

  for (genvar lane = 0; lane < LANES; lane++) begin : gen_lanes
    logic [31:0] copy_a[2**ADDRESS_BITS];
    logic [31:0] copy_b[2**ADDRESS_BITS];
    logic [31:0] copy_c[2**ADDRESS_BITS];

    always_ff @(posedge clock) begin
      if (write_enable[lane]) begin
        copy_a[write_address] <= write_data[lane];
        copy_b[write_address] <= write_data[lane];
        copy_c[write_address] <= write_data[lane];
      end
      if (read) begin
        read_data_a[lane] <= copy_a[read_address_a];
        read_data_b[lane] <= copy_b[read_address_b];
        read_data_c[lane] <= copy_c[read_address_c];
      end
    end
  end

endmodule

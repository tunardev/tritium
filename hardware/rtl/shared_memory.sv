module shared_memory
  import tritium_package::*;
(
    input  logic                   clock,
    input  logic [SHARED_BITS-1:0] address,
    input  logic [            3:0] write_enable,
    input  logic [           31:0] write_data,
    output logic [           31:0] read_data
);

  logic [31:0] words[SHARED_WORDS];

  always_ff @(posedge clock) begin
    for (int byte_index = 0; byte_index < 4; byte_index++)
      if (write_enable[byte_index]) words[address][8*byte_index+:8] <= write_data[8*byte_index+:8];
    read_data <= words[address];
  end

endmodule

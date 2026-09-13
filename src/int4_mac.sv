// The MAC Unit - int4_mac.sv
// Signed INT4 multiplication

`default_nettype none

module int4_mac (
    input  wire signed [3:0] activation,
    input  wire signed [3:0] weight,
    output wire signed [7:0] product
);

    assign product = activation * weight;

endmodule

`default_nettype wire

`default_nettype none
`timescale 1ns / 1ps

// Cocotb wrapper for the CatDog TinyNPU top module.
module tb_tt_um_tinynpu4 ();

  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb_tt_um_tinynpu4);
    #1;
  end

  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  tt_um_tinynpu4 user_project (
      .ui_in(ui_in),
      .uo_out(uo_out),
      .uio_in(uio_in),
      .uio_out(uio_out),
      .uio_oe(uio_oe),
      .ena(ena),
      .clk(clk),
      .rst_n(rst_n)
  );

endmodule

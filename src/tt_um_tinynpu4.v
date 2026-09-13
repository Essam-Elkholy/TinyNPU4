/*
 * Copyright (c) 2026 Essam Elkholy and Team
 * SPDX-License-Identifier: Apache-2.0
 *
 * CatDog TinyNPU - Tiny Tapeout top-level wrapper
 */

`default_nettype none

module tt_um_tinynpu4 (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    wire [3:0] result;
    wire done, result_valid, overflow, accumulator_negative;

    tinynpu4_core u_core (
        .data_in(ui_in), .command(uio_in[2:0]),
        .param_in(uio_in[7:4]), .command_valid(uio_in[3]),
        .clk(clk), .rst_n(rst_n), .enable(ena),
        .result(result), .done(done), .result_valid(result_valid),
        .overflow(overflow), .accumulator_negative(accumulator_negative)
    );

    // result[3:0], done, overflow, accumulator sign, result valid
    assign uo_out = {result_valid, accumulator_negative, overflow, done, result};

    // All bidirectional pins are inputs in this design.
    assign uio_out = 8'b0000_0000;
    assign uio_oe  = 8'b0000_0000;

endmodule

`default_nettype wire

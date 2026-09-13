// The Core Unit  -  tinynpu4_core.v
// Made and tested by Essam Elkholy and Mohamed Awad
// Github link: https://github.com/Essam-Elkholy  "GlitchPi"
//

`default_nettype none

module tinynpu4_core (
    input  wire [7:0] data_in,
    input  wire [2:0] command,
    input  wire [3:0] param_in,
    input  wire       command_valid,

    input  wire       clk,
    input  wire       rst_n,
    input  wire       enable,

    output wire [3:0] result,
    output reg        done,
    output reg        result_valid,
    output reg        overflow,
    output wire       accumulator_negative
);

    // Command definitions
    localparam CMD_NOP             = 3'b000;
    localparam CMD_CLEAR           = 3'b001;
    localparam CMD_LOAD_BIAS_LOW   = 3'b010;
    localparam CMD_LOAD_BIAS_HIGH  = 3'b011;
    localparam CMD_MAC             = 3'b100;
    localparam CMD_FINISH_RELU     = 3'b101;
    localparam CMD_FINISH_LINEAR   = 3'b110;
    localparam CMD_READ_ACC_NIBBLE = 3'b111;

    // State registers for the two parallel lanes
    reg        [3:0]  bias_low_register_0;
    reg        [3:0]  bias_low_register_1;
    reg signed [15:0] accumulator_0;
    reg signed [15:0] accumulator_1;
    reg        [3:0]  result_register;

    // Shared input and lane-select signals
    wire               lane_select;
    wire signed [3:0]  activation;
    wire signed [3:0]  weight_0;
    wire signed [3:0]  weight_1;
    wire signed [15:0] selected_accumulator;

    // Two-lane datapath signals
    wire signed [7:0]  product_0;
    wire signed [7:0]  product_1;
    wire signed [15:0] bias_init_0;
    wire signed [15:0] bias_init_1;
    wire signed [16:0] mac_sum_extended_0;
    wire signed [16:0] mac_sum_extended_1;
    wire        [3:0]  relu_output;
    wire        [3:0]  linear_output;

    // During MAC: one activation is multiplied by two independent weights.
    assign activation = data_in[3:0];
    assign weight_0   = data_in[7:4];
    assign weight_1   = param_in;

    // During bias, finish and debug commands, data_in[0] selects the lane.
    assign lane_select          = data_in[0];
    assign selected_accumulator = lane_select ? accumulator_1 : accumulator_0;

    // Combine both bias nibbles and sign-extend each signed INT8 bias.
    assign bias_init_0 = {{8{param_in[3]}}, param_in, bias_low_register_0};
    assign bias_init_1 = {{8{param_in[3]}}, param_in, bias_low_register_1};

    // Add both signed INT4 products to their 16-bit accumulators.
    assign mac_sum_extended_0 = {accumulator_0[15], accumulator_0}
                              + {{9{product_0[7]}}, product_0};
    assign mac_sum_extended_1 = {accumulator_1[15], accumulator_1}
                              + {{9{product_1[7]}}, product_1};

    // Result outputs
    assign result                = result_register;
    assign accumulator_negative = selected_accumulator[15];

    // First physical signed INT4 multiplier.
    int4_mac u_int4_mac_0 (
        .activation (activation),
        .weight     (weight_0),
        .product    (product_0)
    );

    // Second physical signed INT4 multiplier.
    int4_mac u_int4_mac_1 (
        .activation (activation),
        .weight     (weight_1),
        .product    (product_1)
    );

    // One shared ReLU quantizer, selected between the two accumulators.
    activation_quantizer u_relu_quantizer (
        .accumulator      (selected_accumulator),
        .shift_amount     (param_in),
        .mode_relu        (1'b1),
        .quantized_output (relu_output)
    );

    // One shared Linear quantizer, selected between the two accumulators.
    activation_quantizer u_linear_quantizer (
        .accumulator      (selected_accumulator),
        .shift_amount     (param_in),
        .mode_relu        (1'b0),
        .quantized_output (linear_output)
    );

    // Sequential command controller and two accumulator registers.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_low_register_0 <= 4'd0;
            bias_low_register_1 <= 4'd0;
            accumulator_0       <= 16'sd0;
            accumulator_1       <= 16'sd0;
            result_register     <= 4'd0;
            done                <= 1'b0;
            result_valid        <= 1'b0;
            overflow            <= 1'b0;
        end else begin
            // done is a one-clock pulse.
            done <= 1'b0;

            if (enable && command_valid) begin
                case (command)
                    CMD_NOP: begin
                        // Nothing changes.
                    end

                    CMD_CLEAR: begin
                        bias_low_register_0 <= 4'd0;
                        bias_low_register_1 <= 4'd0;
                        accumulator_0       <= 16'sd0;
                        accumulator_1       <= 16'sd0;
                        result_register     <= 4'd0;
                        result_valid        <= 1'b0;
                        overflow            <= 1'b0;
                    end

                    CMD_LOAD_BIAS_LOW: begin
                        if (lane_select)
                            bias_low_register_1 <= param_in;
                        else
                            bias_low_register_0 <= param_in;
                    end

                    CMD_LOAD_BIAS_HIGH: begin
                        if (lane_select)
                            accumulator_1 <= bias_init_1;
                        else
                            accumulator_0 <= bias_init_0;
                        result_valid <= 1'b0;
                        overflow     <= 1'b0;
                    end

                    CMD_MAC: begin
                        // Lane 0 saturates independently at the signed 16-bit limits.
                        if (mac_sum_extended_0[16] != mac_sum_extended_0[15]) begin
                            overflow <= 1'b1;
                            if (mac_sum_extended_0[16])
                                accumulator_0 <= 16'sh8000;
                            else
                                accumulator_0 <= 16'sd32767;
                        end else begin
                            accumulator_0 <= mac_sum_extended_0[15:0];
                        end

                        // Lane 1 saturates independently at the signed 16-bit limits.
                        if (mac_sum_extended_1[16] != mac_sum_extended_1[15]) begin
                            overflow <= 1'b1;
                            if (mac_sum_extended_1[16])
                                accumulator_1 <= 16'sh8000;
                            else
                                accumulator_1 <= 16'sd32767;
                        end else begin
                            accumulator_1 <= mac_sum_extended_1[15:0];
                        end
                    end

                    CMD_FINISH_RELU: begin
                        result_register <= relu_output;
                        result_valid    <= 1'b1;
                        done            <= 1'b1;
                    end

                    CMD_FINISH_LINEAR: begin
                        result_register <= linear_output;
                        result_valid    <= 1'b1;
                        done            <= 1'b1;
                    end

                    CMD_READ_ACC_NIBBLE: begin
                        // param_in[1:0] selects one of four nibbles in the selected lane.
                        case (param_in[1:0])
                            2'b00: result_register <= selected_accumulator[3:0];
                            2'b01: result_register <= selected_accumulator[7:4];
                            2'b10: result_register <= selected_accumulator[11:8];
                            2'b11: result_register <= selected_accumulator[15:12];
                        endcase
                        result_valid <= 1'b1;
                        done         <= 1'b1;
                    end

                    default: begin
                        // Nothing changes.
                    end
                endcase
            end
        end
    end

endmodule

`default_nettype wire

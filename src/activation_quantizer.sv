// The Activation Unit - activation_quantizer.sv
// ReLU or Linear, arithmetic shift and signed INT4 saturation

`default_nettype none

module activation_quantizer (
    input  wire signed [15:0] accumulator,
    input  wire        [3:0]  shift_amount,
    input  wire               mode_relu,
    output reg         [3:0]  quantized_output
);

    logic signed [15:0] activation_value;
    logic signed [15:0] shifted_value;

    always_comb begin
        // Activation: ReLU or Linear
        if (mode_relu) begin
            if (accumulator < 16'sd0)
                activation_value = 16'sd0;
            else
                activation_value = accumulator;
        end else begin
            activation_value = accumulator;
        end

        // Arithmetic right shift
        shifted_value = activation_value >>> shift_amount;

        // Saturation to signed INT4
        if (shifted_value > 16'sd7)
            quantized_output = 4'sd7;
        else if (shifted_value < -16'sd8)
            quantized_output = -4'sd8;
        else
            quantized_output = shifted_value[3:0];
    end

endmodule

`default_nettype wire

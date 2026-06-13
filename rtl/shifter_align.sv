// shifter_align.sv
//
// Parameterized combinational shifter used to scale fixed-point signals
// back to Q8.8 format by shifting right arithmetically and applying saturation limits.

module shifter_align #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16,
    parameter int SHIFT_AMT = 8
)(
    input  logic signed [IN_WIDTH-1:0]  in_val,
    output logic signed [OUT_WIDTH-1:0] out_val
);

    localparam signed [IN_WIDTH-1:0] OUT_MAX = (1 << (OUT_WIDTH - 1)) - 1;
    localparam signed [IN_WIDTH-1:0] OUT_MIN = -(1 << (OUT_WIDTH - 1));

    logic signed [IN_WIDTH-1:0] shifted;
    assign shifted = in_val >>> SHIFT_AMT;

    always_comb begin
        if (shifted > OUT_MAX) begin
            out_val = OUT_MAX[OUT_WIDTH-1:0];
        end
        else if (shifted < OUT_MIN) begin
            out_val = OUT_MIN[OUT_WIDTH-1:0];
        end
        else begin
            out_val = shifted[OUT_WIDTH-1:0];
        end
    end

endmodule

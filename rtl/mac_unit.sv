// mac_unit
//
// Single-cycle fixed-point multiply-accumulate stage.
//
// The input operands `a` and `b` are signed fixed-point values with
// DATA_WIDTH total bits and FRAC_BITS fractional bits. Their product has twice
// as many fractional bits, so the product is shifted right by FRAC_BITS before
// being added to the 32-bit accumulator input.
//
// Timing:
// - `valid_in` is sampled on the rising edge of `clk`.
// - When `valid_in` is high, `acc_out` is updated with:
//     acc_in + ((a * b) >> FRAC_BITS)
// - `valid_out` is the one-cycle registered version of `valid_in`.
// - `rst` synchronously clears `acc_out` and `valid_out`.
module mac_unit #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 8
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0] a,
    input  logic signed [DATA_WIDTH-1:0] b,
    input  logic signed [31:0] acc_in,
    output logic valid_out,
    output logic signed [31:0] acc_out
);

logic signed [31:0] mult_result;
always_comb begin
    mult_result = a * b;
end

always_ff @(posedge clk) begin
    if (rst) begin
        acc_out   <= 0;
        valid_out <= 0;
    end
    else begin
        valid_out <= valid_in;
        if (valid_in) begin
            acc_out <= acc_in + (mult_result >>> FRAC_BITS);
        end
    end
end
endmodule

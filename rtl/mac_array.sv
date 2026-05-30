module mac_array #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 8,
    parameter INPUTS     = 8,
    parameter OUTPUTS    = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0] activations [INPUTS],
    input  logic signed [DATA_WIDTH-1:0] weights [OUTPUTS][INPUTS],
    output logic signed [31:0] outputs [OUTPUTS],
    output logic valid_out
);

integer i;
integer j;
logic signed [31:0] sum;
logic signed [31:0] scaled_product;

always_ff @(posedge clk) begin
    if (rst) begin
        valid_out <= 0;
        for (i = 0; i < OUTPUTS; i++) begin
            outputs[i] <= 0;
        end
    end
    else begin
        valid_out <= valid_in;
        if (valid_in) begin
            for (i = 0; i < OUTPUTS; i++) begin
                sum = 0;
                for (j = 0; j < INPUTS; j++) begin
                    // Qm.FRAC_BITS operands produce twice as many fractional
                    // bits after multiplication; shift back before summing.
                    scaled_product = (activations[j] * weights[i][j]) >>> FRAC_BITS;
                    sum += scaled_product;
                end
                outputs[i] <= sum;
            end
        end
    end
end
endmodule

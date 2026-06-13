module mac_array #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 8,
    parameter int INPUTS     = 8,
    parameter int OUTPUTS    = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    // activations must be held stable for INPUTS cycles after valid_in
    // is asserted. Back-to-back pipelined vectors are not supported.
    input  logic signed [DATA_WIDTH-1:0]  activations [INPUTS],
    input  logic signed [DATA_WIDTH-1:0] weights [OUTPUTS][INPUTS],
    output logic signed [31:0] outputs [OUTPUTS],
    output logic valid_out
);

logic signed [31:0] partial_sum [OUTPUTS][INPUTS+1];
logic valid_chain[OUTPUTS][INPUTS+1];

genvar o;
genvar i;

generate
for(o=0;o<OUTPUTS;o++) begin : g_OUTPUT_PIPELINE
    assign partial_sum[o][0] = '0;
    assign valid_chain[o][0] = valid_in;
    for(i=0;i<INPUTS;i++) begin : g_MAC_STAGE
        mac_unit #(
            .DATA_WIDTH(DATA_WIDTH),
            .FRAC_BITS(FRAC_BITS)
        ) mac_inst(
            .clk(clk),
            .rst(rst),
            .valid_in(valid_chain[o][i]),
            .a(activations[i]),
            .b(weights[o][i]),
            .acc_in(partial_sum[o][i]),
            .acc_out(partial_sum[o][i+1]),
            .valid_out(valid_chain[o][i+1])
        );
    end
end
endgenerate

integer k;

// Total latency from valid_in to outputs: INPUTS+1 cycles.
// The pipeline itself takes INPUTS cycles to fill; the output
// register adds one more cycle before outputs[] is valid.
assign valid_out = valid_chain[0][INPUTS];
// capture final results
always_ff @(posedge clk) begin
    if (rst) begin
        for (k = 0; k < OUTPUTS; k++) begin
            outputs[k] <= 0;
        end
    end
    else if (valid_out) begin
        for (k = 0; k < OUTPUTS; k++) begin
            outputs[k] <= partial_sum[k][INPUTS];
        end
    end
end

endmodule

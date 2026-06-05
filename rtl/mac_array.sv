module mac_array #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 8,
    parameter int INPUTS     = 8,
    parameter int OUTPUTS    = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0]  activations [INPUTS],
    input  logic signed [DATA_WIDTH-1:0] weights [OUTPUTS][INPUTS],
    output logic signed [31:0] outputs [OUTPUTS],
    output logic valid_out
);

logic signed [31:0] acc_chain [OUTPUTS][INPUTS+1];
logic [INPUTS:0] valid_pipe;

genvar o;
genvar i;

generate

    // Starting accumulator value
    for (o = 0; o < OUTPUTS; o++) begin : g_INIT_ACC
        assign acc_chain[o][0] = 32'sd0;
    end

    // MAC pipelines
    for (o = 0; o < OUTPUTS; o++) begin : g_OUT_GEN
        for (i = 0; i < INPUTS; i++) begin : g_IN_GEN

            logic valid_unused;

            mac_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) mac_inst (
                .clk(clk),
                .rst(rst),

                .valid_in(valid_pipe[i]),

                .a(activations[i]),
                .b(weights[o][i]),

                .acc_in(acc_chain[o][i]),
                .acc_out(acc_chain[o][i+1]),

                .valid_out(valid_unused)
            );
        end
    end

endgenerate


integer k;

// valid pipeline
always_ff @(posedge clk) begin
    if (rst) begin
        valid_pipe <= '0;
    end
    else begin
        valid_pipe[0] <= valid_in;
        for (k = 0; k < INPUTS; k++) begin
            valid_pipe[k+1] <= valid_pipe[k];
        end
    end
end
assign valid_out = valid_pipe[INPUTS];


// capture final results
always_ff @(posedge clk) begin
    if (rst) begin
        for (k = 0; k < OUTPUTS; k++) begin
            outputs[k] <= 0;
        end
    end
    else if (valid_pipe[INPUTS]) begin
        for (k = 0; k < OUTPUTS; k++) begin
            outputs[k] <= acc_chain[k][INPUTS];
        end
    end
end

endmodule

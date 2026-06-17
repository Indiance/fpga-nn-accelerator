module mac_array #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 8,
    parameter int INPUTS     = 8,
    parameter int OUTPUTS    = 4,

    localparam int PROD_WIDTH = 32,
    localparam int LEVELS = $clog2(INPUTS)
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

// reduction tree storage
logic signed [PROD_WIDTH-1:0] tree[OUTPUTS][LEVELS+1][INPUTS];
logic valid_pipe[LEVELS+1];
genvar o,i,l,n;

// parallel multiples
generate

for(o=0;o<OUTPUTS;o++) begin
    for(i=0;i<INPUTS;i++) begin
        always_ff @(posedge clk) begin
            if(rst)
                tree[o][0][i] <= 0;
            else if(valid_in)
                tree[o][0][i] <= ($signed(activations[i]) * $signed(weights[o][i])) >>> FRAC_BITS;
        end
    end
end
endgenerate

// adder tree
generate

for(l=0;l<LEVELS;l++) begin : LEVEL
    localparam int NODES = INPUTS >> (l+1);
    for(o=0;o<OUTPUTS;o++) begin
        for(n=0;n<NODES;n++) begin
            always_ff @(posedge clk) begin
                if(rst)
                    tree[o][l+1][n] <= 0;
                else if(valid_pipe[l])
                    tree[o][l+1][n] <= tree[o][l][2*n] + tree[o][l][2*n+1];
            end
        end
    end
end
endgenerate

integer k;
// valid pipeline
always_ff @(posedge clk) begin
    if(rst)
        valid_pipe <= '{default:0};
    else begin
        valid_pipe[0] <= valid_in;
        for(k=1;k<=LEVELS;k++)
            valid_pipe[k] <= valid_pipe[k-1];
    end

end

assign valid_out = valid_pipe[LEVELS];

// output register
always_ff @(posedge clk) begin
    if(rst) begin
        for(k=0;k<OUTPUTS;k++)
            outputs[k] <= 0;
    end
    else if(valid_pipe[LEVELS]) begin
        for(k=0;k<OUTPUTS;k++)
            outputs[k] <= tree[k][LEVELS][0];
    end
end

endmodule

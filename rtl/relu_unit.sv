module relu_unit #(
    parameter int DATA_WIDTH   = 16,
    parameter int RELU_ENABLE  = 1
)(
    input  logic signed [31:0]          acc_in,
    input  logic signed [DATA_WIDTH-1:0] bias_in,
    output logic signed [DATA_WIDTH-1:0] activated
);

localparam signed [31:0] OUT_MAX = (1 << (DATA_WIDTH - 1)) - 1;
localparam signed [31:0] OUT_MIN = -(1 << (DATA_WIDTH - 1));

logic signed [31:0] biased_val;

always_comb begin
    biased_val = acc_in
               + {{(32 - DATA_WIDTH){bias_in[DATA_WIDTH-1]}}, bias_in};

    if (RELU_ENABLE && biased_val[31]) begin
        activated = '0;
    end
    else if (biased_val > OUT_MAX) begin
        activated = OUT_MAX[DATA_WIDTH-1:0];
    end
    else if (biased_val < OUT_MIN) begin
        activated = OUT_MIN[DATA_WIDTH-1:0];
    end
    else begin
        activated = biased_val[DATA_WIDTH-1:0];
    end
end

endmodule

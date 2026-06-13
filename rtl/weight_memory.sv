module weight_memory #(
    parameter DATA_WIDTH = 16,
    parameter INPUTS = 8,
    parameter OUTPUTS = 4,
    parameter TOTAL_WEIGHTS = 100352, // 784*128
    parameter MEM_FILE = "weights/best_mnist_cnn/classifier_3_weight.mem"
)(
    input logic clk,
    input logic valid_in,
    input logic [$clog2(TOTAL_WEIGHTS)-1:0] base_addr,
    output logic signed [DATA_WIDTH-1:0] weights[OUTPUTS][INPUTS],
    output logic valid_out
);

logic signed [DATA_WIDTH-1:0] mem [0:TOTAL_WEIGHTS-1];

initial
    $readmemh(MEM_FILE, mem);

integer i, j;

always_ff @(posedge clk) begin
    valid_out <= valid_in;
    if (valid_in) begin
        for (i = 0; i < OUTPUTS; i++) begin
            for (j = 0; j < INPUTS; j++) begin
                if (base_addr + i*INPUTS + j < TOTAL_WEIGHTS) begin
                    weights[i][j] <= mem[base_addr + i*INPUTS + j];
                end
                else begin
                    weights[i][j] <= 0;
                end
            end
        end
    end
end

endmodule
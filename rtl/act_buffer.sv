module act_buffer #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH = 128,
    parameter int READ_WIDTH = 8
)(
    input logic clk,
    input logic rst,

    // write side
    input logic write_en,
    input logic [$clog2(DEPTH)-1:0] write_addr,
    input logic signed [DATA_WIDTH-1:0] write_data,

    // read side
    input logic read_en,
    input logic [$clog2(DEPTH)-1:0] read_addr,
    output logic signed [DATA_WIDTH-1:0] activations [READ_WIDTH]
);

logic signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

integer i;

always_ff @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < READ_WIDTH; i++) begin
            activations[i] <= 0;
        end
    end
    else begin

        // write
        if (write_en) begin
            if (write_addr < DEPTH) begin
                mem[write_addr] <= write_data;
            end
        end

        // read 8 activations at once
        if (read_en) begin
            for (i = 0; i < READ_WIDTH; i++) begin
                if (read_addr + i < DEPTH) begin
                    activations[i] <= mem[read_addr + i];
                end
                else begin
                    activations[i] <= 0;
                end
            end
        end
    end
end

endmodule


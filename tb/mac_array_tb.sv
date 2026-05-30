`timescale 1ns/1ps
module tb_mac_array;
logic clk;
logic rst;
logic valid_in;
logic valid_out;
logic signed [15:0] activations [2];
logic signed [15:0] weights [2][2];
logic signed [31:0] outputs [2];

mac_array #(
    .INPUTS(2),
    .OUTPUTS(2)
) dut (
    .clk(clk),
    .rst(rst),
    .valid_in(valid_in),
    .activations(activations),
    .weights(weights),
    .outputs(outputs),
    .valid_out(valid_out)
);

always #5 clk = ~clk;
initial begin
    clk = 0;
    rst = 1;
    valid_in = 0;
    activations[0] = 0;
    activations[1] = 0;
    weights[0][0] = 0;
    weights[0][1] = 0;
    weights[1][0] = 0;
    weights[1][1] = 0;

    #20;
    rst = 0;

    // Q8.8 inputs: activations = [1.0, 2.0].
    activations[0] = 16'sd256;
    activations[1] = 16'sd512;

    // Q8.8 weights:
    // [ [1.0, 2.0],
    //   [3.0, 4.0] ]
    weights[0][0] = 16'sd256;
    weights[0][1] = 16'sd512;
    weights[1][0] = 16'sd768;
    weights[1][1] = 16'sd1024;

    valid_in = 1;
    #10;
    valid_in = 0;
    #10;

    // Expected dot products in Q8.8:
    // output0 = (1.0 * 1.0) + (2.0 * 2.0) = 5.0  -> 1280
    // output1 = (1.0 * 3.0) + (2.0 * 4.0) = 11.0 -> 2816
    if (outputs[0] == 1280)
        $display("PASS output0");
    else
        $error("FAIL output0 expected 1280 got %0d", outputs[0]);
    if (outputs[1] == 2816)
        $display("PASS output1");
    else
        $error("FAIL output1 expected 2816 got %0d", outputs[1]);
    $display("ALL TESTS PASSED");
    $finish;
end
endmodule

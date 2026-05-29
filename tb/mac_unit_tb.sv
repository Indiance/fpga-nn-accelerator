`timescale 1ns/1ps

module tb_mac_unit;

logic clk;
logic rst;
logic valid_in;

logic signed [15:0] a;
logic signed [15:0] b;

logic signed [31:0] acc_in;
logic signed [31:0] acc_out;

logic valid_out;

mac_unit dut (
    .clk(clk),
    .rst(rst),
    .valid_in(valid_in),
    .a(a),
    .b(b),
    .acc_in(acc_in),
    .acc_out(acc_out),
    .valid_out(valid_out)
);

always #5 clk = ~clk;
initial begin
    clk = 0;
    rst = 1;
    valid_in = 0;
    a = 0;
    b = 0;
    acc_in = 0;

    #20;
    rst = 0;

    //--------------------------------------------------
    // TEST 1: 1.5 * 2.0 = 3.0
    //--------------------------------------------------
    a = 16'sd384;      // 1.5 * 256
    b = 16'sd512;      // 2.0 * 256
    acc_in = 0;
    valid_in = 1;

    #10;
    valid_in = 0;

    #10;

    if (acc_out == 32'sd768)
        $display("PASS: TEST1");
    else
        $error("FAIL: TEST1 Expected 768 Got %0d", acc_out);

    //--------------------------------------------------
    // TEST 2: -1.5 * 2.0 = -3.0
    //--------------------------------------------------
    a = -16'sd384;
    b = 16'sd512;
    acc_in = 0;
    valid_in = 1;

    #10;
    valid_in = 0;

    #10;

    if (acc_out == -32'sd768)
        $display("PASS: TEST2");
    else
        $error("FAIL: TEST2 Expected -768 Got %0d", acc_out);

    //--------------------------------------------------
    // TEST 3: Accumulation
    // 1.0 + (1.5 * 2.0) = 4.0
    //--------------------------------------------------
    a = 16'sd384;
    b = 16'sd512;
    acc_in = 32'sd256;    // 1.0
    valid_in = 1;

    #10;
    valid_in = 0;

    #10;

    if (acc_out == 32'sd1024)
        $display("PASS: TEST3");
    else
        $error("FAIL: TEST3 Expected 1024 Got %0d", acc_out);

    //--------------------------------------------------
    // TEST 4: Zero
    //--------------------------------------------------
    a = 0;
    b = 16'sd512;
    acc_in = 0;
    valid_in = 1;

    #10;
    valid_in = 0;

    #10;

    if (acc_out == 0)
        $display("PASS: TEST4");
    else
        $error("FAIL: TEST4 Expected 0 Got %0d", acc_out);

    $display("ALL TESTS COMPLETE");
    $finish;
end

endmodule

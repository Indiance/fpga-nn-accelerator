`timescale 1ns/1ps

module tb_relu_unit;

    parameter int DATA_WIDTH = 16;

    // DUT 1 signals (ReLU Enabled)
    logic signed [31:0]           acc_in_1;
    logic signed [DATA_WIDTH-1:0] bias_in_1;
    logic signed [DATA_WIDTH-1:0] activated_1;

    // DUT 2 signals (ReLU Disabled)
    logic signed [31:0]           acc_in_2;
    logic signed [DATA_WIDTH-1:0] bias_in_2;
    logic signed [DATA_WIDTH-1:0] activated_2;

    // Instantiate DUT with ReLU enabled (default)
    relu_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .RELU_ENABLE(1)
    ) dut_relu_en (
        .acc_in(acc_in_1),
        .bias_in(bias_in_1),
        .activated(activated_1)
    );

    // Instantiate DUT with ReLU disabled
    relu_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .RELU_ENABLE(0)
    ) dut_relu_ds (
        .acc_in(acc_in_2),
        .bias_in(bias_in_2),
        .activated(activated_2)
    );

    initial begin
        $display("========================================");
        $display("Starting tb_relu_unit simulation...");
        $display("========================================");

        // ----------------------------------------------------
        // TEST SECTION 1: ReLU Enabled (RELU_ENABLE = 1)
        // ----------------------------------------------------
        
        // Test 1.1: Positive inputs within range -> unchanged
        acc_in_1  = 32'sd1000;
        bias_in_1 = 16'sd500;
        #1;
        if (activated_1 === 16'sd1500)
            $display("PASS: Test 1.1 (Positive within range) - Got: %0d", activated_1);
        else
            $error("FAIL: Test 1.1 (Positive within range) - Expected: 1500, Got: %0d", activated_1);

        // Test 1.2: Negative inputs -> 0
        acc_in_1  = -32'sd1000;
        bias_in_1 = 16'sd200; // Biased = -800
        #1;
        if (activated_1 === 16'sd0)
            $display("PASS: Test 1.2 (Negative to zero) - Got: %0d", activated_1);
        else
            $error("FAIL: Test 1.2 (Negative to zero) - Expected: 0, Got: %0d", activated_1);

        // Test 1.3: Positive overflow saturation -> OUT_MAX (32767)
        acc_in_1  = 32'sd40000;
        bias_in_1 = 16'sd1000; // Biased = 41000
        #1;
        if (activated_1 === 16'sd32767)
            $display("PASS: Test 1.3 (Positive overflow saturation) - Got: %0d", activated_1);
        else
            $error("FAIL: Test 1.3 (Positive overflow saturation) - Expected: 32767, Got: %0d", activated_1);

        // Test 1.4: Extreme negative input with ReLU -> 0
        acc_in_1  = -32'sd50000;
        bias_in_1 = -16'sd5000; // Biased = -55000
        #1;
        if (activated_1 === 16'sd0)
            $display("PASS: Test 1.4 (Extreme negative) - Got: %0d", activated_1);
        else
            $error("FAIL: Test 1.4 (Extreme negative) - Expected: 0, Got: %0d", activated_1);

        // ----------------------------------------------------
        // TEST SECTION 2: ReLU Disabled (RELU_ENABLE = 0)
        // ----------------------------------------------------
        
        // Test 2.1: Positive inputs within range -> unchanged
        acc_in_2  = 32'sd2000;
        bias_in_2 = -16'sd500; // Biased = 1500
        #1;
        if (activated_2 === 16'sd1500)
            $display("PASS: Test 2.1 (No ReLU, Positive within range) - Got: %0d", activated_2);
        else
            $error("FAIL: Test 2.1 (No ReLU, Positive within range) - Expected: 1500, Got: %0d", activated_2);

        // Test 2.2: Negative inputs within range -> unchanged
        acc_in_2  = -32'sd3000;
        bias_in_2 = 16'sd500; // Biased = -2500
        #1;
        if (activated_2 === -16'sd2500)
            $display("PASS: Test 2.2 (No ReLU, Negative within range) - Got: %0d", activated_2);
        else
            $error("FAIL: Test 2.2 (No ReLU, Negative within range) - Expected: -2500, Got: %0d", activated_2);

        // Test 2.3: Positive overflow saturation -> OUT_MAX (32767)
        acc_in_2  = 32'sd35000;
        bias_in_2 = 16'sd0; // Biased = 35000
        #1;
        if (activated_2 === 16'sd32767)
            $display("PASS: Test 2.3 (No ReLU, Positive overflow) - Got: %0d", activated_2);
        else
            $error("FAIL: Test 2.3 (No ReLU, Positive overflow) - Expected: 32767, Got: %0d", activated_2);

        // Test 2.4: Negative underflow saturation -> OUT_MIN (-32768)
        acc_in_2  = -32'sd35000;
        bias_in_2 = -16'sd500; // Biased = -35500
        #1;
        if (activated_2 === -16'sd32768)
            $display("PASS: Test 2.4 (No ReLU, Negative underflow) - Got: %0d", activated_2);
        else
            $error("FAIL: Test 2.4 (No ReLU, Negative underflow) - Expected: -32768, Got: %0d", activated_2);

        $display("========================================");
        $display("tb_relu_unit simulation finished!");
        $display("========================================");
        $finish;
    end

endmodule

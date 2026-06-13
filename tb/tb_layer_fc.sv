`timescale 1ns/1ps

module tb_layer_fc;

    // Simulation parameters matching python/fixed_point_infer.py configuration
    parameter int DATA_WIDTH  = 16;
    parameter int FRAC_BITS   = 8;
    parameter int INPUT_SIZE  = 16;
    parameter int OUTPUT_SIZE = 8;
    parameter int INPUTS      = 4;
    parameter int OUTPUTS     = 2;
    parameter int RELU_ENABLE  = 1;
    parameter     WEIGHT_FILE  = "tb_weights.mem";
    parameter     BIAS_FILE    = "tb_biases.mem";

    logic clk;
    logic rst;
    logic start;
    logic done;

    logic                          act_wr_en;
    logic [$clog2(INPUT_SIZE)-1:0]  act_wr_addr;
    logic signed [DATA_WIDTH-1:0]   act_wr_data;

    logic [$clog2(OUTPUT_SIZE)-1:0]  act_rd_addr;
    logic signed [DATA_WIDTH-1:0]    act_rd_data;

    // Golden reference array buffers
    logic signed [DATA_WIDTH-1:0] test_inputs [INPUT_SIZE];
    logic signed [DATA_WIDTH-1:0] expected_outputs [OUTPUT_SIZE];

    // Instantiate DUT
    layer_fc #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .INPUTS(INPUTS),
        .OUTPUTS(OUTPUTS),
        .RELU_ENABLE(RELU_ENABLE),
        .WEIGHT_FILE(WEIGHT_FILE),
        .BIAS_FILE(BIAS_FILE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .act_wr_en(act_wr_en),
        .act_wr_addr(act_wr_addr),
        .act_wr_data(act_wr_data),
        .act_rd_addr(act_rd_addr),
        .act_rd_data(act_rd_data)
    );

    // Clock generator (100 MHz)
    always #5 clk = ~clk;

    initial begin
        // Load test vectors
        $readmemh("tb_inputs.mem", test_inputs);
        $readmemh("tb_expected_outputs.mem", expected_outputs);

        $display("========================================");
        $display("Starting tb_layer_fc simulation...");
        $display("========================================");

        // Initialize signals
        clk = 0;
        rst = 1;
        start = 0;
        act_wr_en = 0;
        act_wr_addr = 0;
        act_wr_data = 0;
        act_rd_addr = 0;

        #20;
        rst = 0;
        #10;

        // 1. Write inputs into input buffer
        $display("Loading inputs into act_buffer...");
        for (int i = 0; i < INPUT_SIZE; i++) begin
            @(posedge clk);
            act_wr_en   = 1;
            act_wr_addr = i;
            act_wr_data = test_inputs[i];
        end
        @(posedge clk);
        act_wr_en = 0;

        // 2. Start layer processing
        $display("Starting layer computation...");
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // 3. Wait for done assertion
        $display("Waiting for done signal...");
        @(posedge done);
        $display("Done signal asserted! Verification in progress...");

        // Wait an extra clock cycle to align with registered read
        @(posedge clk);

        // 4. Verify outputs against reference
        for (int o = 0; o < OUTPUT_SIZE; o++) begin
            act_rd_addr = o;
            @(posedge clk); // Wait 1 cycle for registered read output
            #1; // Wait a tiny step for value to stabilize
            if (act_rd_data === expected_outputs[o]) begin
                $display("PASS: Output %0d - Got: %0d (0x%h)", o, act_rd_data, act_rd_data);
            end else begin
                $error("FAIL: Output %0d - Expected: %0d (0x%h), Got: %0d (0x%h)", 
                       o, expected_outputs[o], expected_outputs[o], act_rd_data, act_rd_data);
            end
        end

        $display("========================================");
        $display("tb_layer_fc simulation finished!");
        $display("========================================");
        $finish;
    end

endmodule

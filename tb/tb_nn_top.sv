`timescale 1ns/1ps

module tb_nn_top;

    parameter int DATA_WIDTH = 16;
    parameter int FRAC_BITS  = 8;

    logic clk;
    logic rst_n;
    logic infer_start;
    logic signed [DATA_WIDTH-1:0] pixel_data [783:0];

    logic infer_done;
    logic [3:0] class_out;

    // Buffer to hold 100 test images (each is 784 words)
    logic signed [DATA_WIDTH-1:0] test_images [105 * 784];
    // Buffer to hold 100 expected classes
    logic [3:0] expected_classes [0:104];

    // Instantiate DUT
    nn_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .infer_start(infer_start),
        .pixel_data(pixel_data),
        .infer_done(infer_done),
        .class_out(class_out)
    );

    // Clock generator (100 MHz)
    always #5 clk = ~clk;

    int correct_cnt;
    int img_idx;

    initial begin
        // Load test images and labels
        $readmemh("tb_input_images.mem", test_images);
        $readmemh("tb_expected_classes.mem", expected_classes);

        // Diagnostics checking if weights/biases loaded correctly
        #1;
        $display("[Diagnostic] FC1 Weight 0: %h (expected non-zero, non-x)", dut.u_fc1.u_weight_mem.mem[0]);
        $display("[Diagnostic] FC1 Bias 0: %h", dut.u_fc1.bias_mem[0]);
        $display("[Diagnostic] FC3 Weight 0: %h", dut.u_fc3.u_weight_mem.mem[0]);
        $display("[Diagnostic] FC3 Bias 0: %h", dut.u_fc3.bias_mem[0]);
        if (dut.u_fc1.u_weight_mem.mem[0] === 'x || dut.u_fc1.u_weight_mem.mem[0] === '0) begin
            $warning("[Diagnostic Warning] FC1 weights are uninitialized or zero! Verify weights/fc1_weights.mem search path.");
        end

        $display("==================================================");
        $display("Starting tb_nn_top MNIST Inference Verification...");
        $display("==================================================");

        // Initialize signals
        clk = 0;
        rst_n = 0;
        infer_start = 0;
        correct_cnt = 0;
        
        for (int i = 0; i < 784; i++) begin
            pixel_data[i] = 0;
        end

        // Apply reset
        #20;
        rst_n = 1;
        #20;

        // Loop over all 100 test images
        for (img_idx = 0; img_idx < 100; img_idx++) begin
            // Load current image into pixel_data port
            @(posedge clk);
            for (int p = 0; p < 784; p++) begin
                pixel_data[p] = test_images[img_idx * 784 + p];
            end

            // Assert infer_start for 1 cycle
            infer_start = 1;
            @(posedge clk);
            infer_start = 0;

            // Wait for infer_done to be asserted
            @(posedge infer_done);

            // Compare prediction against label
            if (class_out === expected_classes[img_idx]) begin
                correct_cnt++;
                if (img_idx % 10 == 0 || img_idx == 99) begin
                    $display("Image %3d - PASS (Prediction: %0d, Label: %0d) [Running accuracy: %0d/%0d]", 
                             img_idx, class_out, expected_classes[img_idx], correct_cnt, img_idx + 1);
                end
            end
            else begin
                if (img_idx % 10 == 0 || img_idx == 99) begin
                    $display("Image %3d - FAIL (Prediction: %0d, Label: %0d) [Running accuracy: %0d/%0d]", 
                             img_idx, class_out, expected_classes[img_idx], correct_cnt, img_idx + 1);
                end
            end
            
            // Wait a few cycles before the next image inference
            #20;
        end

        // Report final accuracy
        $display("==================================================");
        $display("Inference Complete!");
        $display("Total correct predictions: %0d / 100", correct_cnt);
        $display("Overall accuracy: %0d%%", correct_cnt);
        $display("==================================================");

        if (correct_cnt >= 90) begin
            $display("SUCCESS: Top-level classification accuracy (>= 90%%) meets target requirements.");
        end
        else begin
            $error("ERROR: Top-level classification accuracy is below the 90%% requirement.");
        end
        $display("==================================================");

        $finish;
    end

endmodule

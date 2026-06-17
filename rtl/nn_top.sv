// nn_top.sv
//
// Top-level MNIST accelerator wrapper.
// Instantiates and connects three fully-connected layers (FC1, FC2, FC3) in series,
// utilizing an internal state machine to copy intermediate activations.
// Computes argmax of the final layer outputs to predict the digit class.

module nn_top #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 8
)(
    input  logic clk,
    input  logic rst_n,

    input  logic infer_start,
    input  logic signed [DATA_WIDTH-1:0] pixel_data [783:0],

    output logic infer_done,
    output logic [3:0] class_out
);

    // Active-high synchronous reset derived from active-low reset
    logic rst;
    assign rst = ~rst_n;

    // Layer 1 (FC1) Signals
    logic        fc1_start;
    logic        fc1_done;
    logic        fc1_act_wr_en;
    logic [9:0]  fc1_act_wr_addr;
    logic signed [DATA_WIDTH-1:0] fc1_act_wr_data;
    logic [6:0]  fc1_act_rd_addr;
    logic signed [DATA_WIDTH-1:0] fc1_act_rd_data;

    // Layer 2 (FC2) Signals
    logic        fc2_start;
    logic        fc2_done;
    logic        fc2_act_wr_en;
    logic [6:0]  fc2_act_wr_addr;
    logic signed [DATA_WIDTH-1:0] fc2_act_wr_data;
    logic [5:0]  fc2_act_rd_addr;
    logic signed [DATA_WIDTH-1:0] fc2_act_rd_data;

    // Layer 3 (FC3) Signals
    logic        fc3_start;
    logic        fc3_done;
    logic        fc3_act_wr_en;
    logic [5:0]  fc3_act_wr_addr;
    logic signed [DATA_WIDTH-1:0] fc3_act_wr_data;
    logic [3:0]  fc3_act_rd_addr;
    logic signed [DATA_WIDTH-1:0] fc3_act_rd_data;

    // Register file for storing final FC3 layer outputs
    logic signed [DATA_WIDTH-1:0] fc3_out_reg [9:0];

    // FSM States
    typedef enum logic [3:0] {
        IDLE,
        LOAD_FC1,
        FC1_START,
        FC1_WAIT,
        COPY_FC1_FC2,
        FC2_START,
        FC2_WAIT,
        COPY_FC2_FC3,
        FC3_START,
        FC3_WAIT,
        READ_FC3,
        ARGMAX,
        DONE
    } state_t;

    state_t state;
    logic [9:0] cnt; // General counter for loading/copying states

    // Instantiation of FC1: 784 -> 128 (INPUTS=8, OUTPUTS=8, ReLU Enabled)
    layer_fc #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .INPUT_SIZE(784),
        .OUTPUT_SIZE(128),
        .INPUTS(8),
        .OUTPUTS(8),
        .RELU_ENABLE(1),
        .WEIGHT_FILE("weights/fc1_weights.mem"),
        .BIAS_FILE("weights/fc1_bias.mem")
    ) u_fc1 (
        .clk(clk),
        .rst(rst),
        .start(fc1_start),
        .done(fc1_done),
        .act_wr_en(fc1_act_wr_en),
        .act_wr_addr(fc1_act_wr_addr),
        .act_wr_data(fc1_act_wr_data),
        .act_rd_addr(fc1_act_rd_addr),
        .act_rd_data(fc1_act_rd_data)
    );

    // Instantiation of FC2: 128 -> 64 (INPUTS=4, OUTPUTS=4, ReLU Enabled)
    layer_fc #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .INPUT_SIZE(128),
        .OUTPUT_SIZE(64),
        .INPUTS(4),
        .OUTPUTS(4),
        .RELU_ENABLE(1),
        .WEIGHT_FILE("weights/fc2_weights.mem"),
        .BIAS_FILE("weights/fc2_bias.mem")
    ) u_fc2 (
        .clk(clk),
        .rst(rst),
        .start(fc2_start),
        .done(fc2_done),
        .act_wr_en(fc2_act_wr_en),
        .act_wr_addr(fc2_act_wr_addr),
        .act_wr_data(fc2_act_wr_data),
        .act_rd_addr(fc2_act_rd_addr),
        .act_rd_data(fc2_act_rd_data)
    );

    // Instantiation of FC3: 64 -> 10 (INPUTS=4, OUTPUTS=2, ReLU Disabled)
    layer_fc #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .INPUT_SIZE(64),
        .OUTPUT_SIZE(10),
        .INPUTS(4),
        .OUTPUTS(2),
        .RELU_ENABLE(0),
        .WEIGHT_FILE("weights/fc3_weights.mem"),
        .BIAS_FILE("weights/fc3_bias.mem")
    ) u_fc3 (
        .clk(clk),
        .rst(rst),
        .start(fc3_start),
        .done(fc3_done),
        .act_wr_en(fc3_act_wr_en),
        .act_wr_addr(fc3_act_wr_addr),
        .act_wr_data(fc3_act_wr_data),
        .act_rd_addr(fc3_act_rd_addr),
        .act_rd_data(fc3_act_rd_data)
    );

    // Bounded write address for fc3_out_reg to prevent out-of-bounds compilation/simulation crashes
    logic [3:0] fc3_write_addr;
    always_comb begin
        if (cnt >= 2 && cnt <= 11) begin
            fc3_write_addr = cnt[3:0] - 4'd2;
        end
        else begin
            fc3_write_addr = 4'd0;
        end
    end

    // Argmax logic: find digit class with the largest activation
    logic signed [DATA_WIDTH-1:0] max_val;
    logic [3:0] max_idx;

    always_comb begin
        max_val = fc3_out_reg[0];
        max_idx = 0;
        for (int i = 1; i < 10; i++) begin
            if (fc3_out_reg[i] > max_val) begin
                max_val = fc3_out_reg[i];
                max_idx = i[3:0];
            end
        end
    end

    // Sequential state transitions and control path sequencing
    always_ff @(posedge clk) begin
        if (rst) begin
            state           <= IDLE;
            cnt             <= 0;
            fc1_start       <= 0;
            fc2_start       <= 0;
            fc3_start       <= 0;
            fc1_act_wr_en   <= 0;
            fc1_act_wr_addr <= 0;
            fc1_act_wr_data <= 0;
            fc2_act_wr_en   <= 0;
            fc2_act_wr_addr <= 0;
            fc2_act_wr_data <= 0;
            fc3_act_wr_en   <= 0;
            fc3_act_wr_addr <= 0;
            fc3_act_wr_data <= 0;
            fc1_act_rd_addr <= 0;
            fc2_act_rd_addr <= 0;
            fc3_act_rd_addr <= 0;
            infer_done      <= 0;
            class_out       <= 0;
            for (int i = 0; i < 10; i++) begin
                fc3_out_reg[i] <= 0;
            end
        end
        else begin
            case (state)
                IDLE: begin
                    infer_done <= 0;
                    if (infer_start) begin
                        cnt   <= 0;
                        state <= LOAD_FC1;
                    end
                end

                // Load 784 pixel words into FC1 input activation buffer
                LOAD_FC1: begin
                    fc1_act_wr_en   <= 1;
                    fc1_act_wr_addr <= cnt;
                    fc1_act_wr_data <= pixel_data[cnt];
                    if (cnt == 783) begin
                        cnt   <= 0;
                        state <= FC1_START;
                    end
                    else begin
                        cnt   <= cnt + 1;
                    end
                end

                FC1_START: begin
                    fc1_act_wr_en <= 0;
                    fc1_start     <= 1;
                    state         <= FC1_WAIT;
                end

                FC1_WAIT: begin
                    fc1_start <= 0;
                    if (fc1_done) begin
                        cnt   <= 0;
                        state <= COPY_FC1_FC2;
                    end
                end

                // Pipelined copy of 128 elements from FC1 to FC2
                COPY_FC1_FC2: begin
                    if (cnt < 128) begin
                        fc1_act_rd_addr <= cnt[6:0];
                    end
                    else begin
                        fc1_act_rd_addr <= 7'd0;
                    end
                    if (cnt >= 2) begin
                        fc2_act_wr_en   <= 1;
                        fc2_act_wr_addr <= cnt[6:0] - 2;
                        fc2_act_wr_data <= fc1_act_rd_data;
                    end
                    if (cnt == 129) begin
                        cnt             <= 0;
                        state           <= FC2_START;
                    end
                    else begin
                        cnt <= cnt + 1;
                    end
                end

                FC2_START: begin
                    fc2_act_wr_en <= 0;
                    fc2_start     <= 1;
                    state         <= FC2_WAIT;
                end

                FC2_WAIT: begin
                    fc2_start <= 0;
                    if (fc2_done) begin
                        cnt   <= 0;
                        state <= COPY_FC2_FC3;
                    end
                end

                // Pipelined copy of 64 elements from FC2 to FC3
                COPY_FC2_FC3: begin
                    if (cnt < 64) begin
                        fc2_act_rd_addr <= cnt[5:0];
                    end
                    else begin
                        fc2_act_rd_addr <= 6'd0;
                    end
                    if (cnt >= 2) begin
                        fc3_act_wr_en   <= 1;
                        fc3_act_wr_addr <= cnt[5:0] - 2;
                        fc3_act_wr_data <= fc2_act_rd_data;
                    end
                    if (cnt == 65) begin
                        cnt             <= 0;
                        state           <= FC3_START;
                    end
                    else begin
                        cnt <= cnt + 1;
                    end
                end

                FC3_START: begin
                    fc3_act_wr_en <= 0;
                    fc3_start     <= 1;
                    state         <= FC3_WAIT;
                end

                FC3_WAIT: begin
                    fc3_start <= 0;
                    if (fc3_done) begin
                        cnt   <= 0;
                        state <= READ_FC3;
                    end
                end

                // Pipelined read of 10 elements from FC3
                READ_FC3: begin
                    if (cnt < 10) begin
                        fc3_act_rd_addr <= cnt[3:0];
                    end
                    else begin
                        fc3_act_rd_addr <= 4'd0;
                    end
                    if (cnt >= 2) begin
                        fc3_out_reg[fc3_write_addr] <= fc3_act_rd_data;
                    end
                    if (cnt == 11) begin
                        cnt   <= 0;
                        state <= ARGMAX;
                    end
                    else begin
                        cnt <= cnt + 1;
                    end
                end

                // Predict classification class index in 1 cycle
                ARGMAX: begin
                    class_out <= max_idx;
                    state     <= DONE;
                end

                // Assert infer_done for 1 cycle
                DONE: begin
                    infer_done <= 1;
                    state      <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

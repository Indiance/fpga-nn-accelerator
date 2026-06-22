// layer_fc.sv
//
// Fully-connected layer datapath:
//   Y[o] = activation(sum_i(W[o][i] * X[i]) + b[o])
//
// Control sequencing is handled by fsm_controller.sv.
// Post-processing is handled by relu_unit.sv.
//
// The INPUT_SIZE x OUTPUT_SIZE dot product is tiled over the mac_array:
//   IN_TILES  = INPUT_SIZE  / INPUTS   (input-dimension tiles)
//   OUT_TILES = OUTPUT_SIZE / OUTPUTS  (output-dimension tiles)
//
// Weight file format (WEIGHT_FILE):
//   Tiled hex file. For tile (out_tile, in_tile) the OUTPUTS*INPUTS weights
//   begin at address (out_tile * IN_TILES + in_tile) * OUTPUTS * INPUTS.
//   Within a tile, weights are row-major by output then input:
//     w[o][j] at offset o * INPUTS + j
//
// Bias file format (BIAS_FILE):
//   Hex file, one DATA_WIDTH-bit word per output neuron.

module layer_fc #(
    parameter int DATA_WIDTH  = 16,
    parameter int FRAC_BITS   = 8,
    parameter int INPUT_SIZE  = 784,
    parameter int OUTPUT_SIZE = 128,
    parameter int INPUTS      = 8,
    parameter int OUTPUTS     = 4,
    parameter int RELU_ENABLE  = 1,
    parameter     WEIGHT_FILE  = "weights.mem",
    parameter     BIAS_FILE    = "biases.mem"
)(
    input  logic clk,
    input  logic rst,

    input  logic start,
    output logic done,

    input  logic                          act_wr_en,
    input  logic [$clog2(INPUT_SIZE)-1:0]  act_wr_addr,
    input  logic signed [DATA_WIDTH-1:0]   act_wr_data,

    input  logic [$clog2(OUTPUT_SIZE)-1:0]  act_rd_addr,
    output logic signed [DATA_WIDTH-1:0]    act_rd_data
);

localparam int IN_TILES  = INPUT_SIZE / INPUTS;
localparam int OUT_TILES = OUTPUT_SIZE / OUTPUTS;

logic signed [DATA_WIDTH-1:0] bias_mem [OUTPUT_SIZE];
initial $readmemh(BIAS_FILE, bias_mem);

logic signed [DATA_WIDTH-1:0] act_out_buf [OUTPUT_SIZE];
logic signed [31:0]           acc_buf     [OUTPUT_SIZE];

logic                          fetch_en;
logic                          postproc_en;
logic [$clog2(IN_TILES + 1)-1:0]  in_tile;
logic [$clog2(OUT_TILES + 1)-1:0] out_tile;
logic [$clog2(OUTPUTS + 1)-1:0]    pp_cnt;

logic                          ab_read_en;
logic [$clog2(INPUT_SIZE)-1:0] ab_read_addr;
logic signed [DATA_WIDTH-1:0]  ab_activations [INPUTS];

logic                                      wm_valid_in;
logic [$clog2(OUTPUT_SIZE * INPUT_SIZE)-1:0] wm_base_addr;
logic signed [DATA_WIDTH-1:0]              wm_weights [OUTPUTS][INPUTS];
logic                                      wm_valid_out;

logic signed [31:0] mac_outputs [OUTPUTS];
logic               mac_valid_out;
logic               mac_valid_out_r;

logic signed [DATA_WIDTH-1:0] activated_val;

act_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEPTH     (INPUT_SIZE),
    .READ_WIDTH(INPUTS)
) u_act_in (
    .clk        (clk),
    .rst        (rst),
    .write_en   (act_wr_en),
    .write_addr (act_wr_addr),
    .write_data (act_wr_data),
    .read_en    (ab_read_en),
    .read_addr  (ab_read_addr),
    .activations(ab_activations)
);

weight_memory #(
    .DATA_WIDTH   (DATA_WIDTH),
    .INPUTS       (INPUTS),
    .OUTPUTS      (OUTPUTS),
    .TOTAL_WEIGHTS(OUTPUT_SIZE * INPUT_SIZE),
    .MEM_FILE     (WEIGHT_FILE)
) u_weight_mem (
    .clk      (clk),
    .valid_in (wm_valid_in),
    .base_addr(wm_base_addr),
    .weights  (wm_weights),
    .valid_out(wm_valid_out)
);

mac_array #(
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_BITS (FRAC_BITS),
    .INPUTS    (INPUTS),
    .OUTPUTS   (OUTPUTS)
) u_mac_array (
    .clk        (clk),
    .rst        (rst),
    .valid_in   (wm_valid_out),
    .activations(ab_activations),
    .weights    (wm_weights),
    .outputs    (mac_outputs),
    .valid_out  (mac_valid_out)
);

fsm_controller #(
    .INPUT_SIZE (INPUT_SIZE),
    .OUTPUT_SIZE(OUTPUT_SIZE),
    .INPUTS     (INPUTS),
    .OUTPUTS    (OUTPUTS),
    .PIPELINE_LATENCY($clog2(INPUTS) + 3)
) u_fsm_controller (
    .clk            (clk),
    .rst            (rst),
    .start          (start),
    .done           (done),
    .fetch_en       (fetch_en),
    .postproc_en    (postproc_en),
    .in_tile        (in_tile),
    .out_tile       (out_tile),
    .pp_cnt         (pp_cnt)
);

logic [$clog2(OUTPUT_SIZE)-1:0] safe_pp_idx;
always_comb begin
    if (out_tile * OUTPUTS + pp_cnt < OUTPUT_SIZE) begin
        safe_pp_idx = out_tile * OUTPUTS + pp_cnt;
    end
    else begin
        safe_pp_idx = '0;
    end
end

relu_unit #(
    .DATA_WIDTH (DATA_WIDTH),
    .RELU_ENABLE(RELU_ENABLE)
) u_relu_unit (
    .acc_in   (acc_buf[safe_pp_idx]),
    .bias_in  (bias_mem[safe_pp_idx]),
    .activated(activated_val)
);

always_ff @(posedge clk) begin
    if (rst)
        mac_valid_out_r <= '0;
    else
        mac_valid_out_r <= mac_valid_out;
end

assign ab_read_en   = fetch_en;
assign wm_valid_in  = fetch_en;
assign ab_read_addr = in_tile * INPUTS;
assign wm_base_addr = (out_tile * IN_TILES + in_tile) * (OUTPUTS * INPUTS);

always_ff @(posedge clk) begin
    if (rst) begin
        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            acc_buf[i]     <= '0;
            act_out_buf[i] <= '0;
        end
    end
    else begin
        if (fetch_en && (in_tile == '0)) begin
            for (int i = 0; i < OUTPUTS; i++) begin
                if (out_tile * OUTPUTS + i < OUTPUT_SIZE) begin
                    acc_buf[out_tile * OUTPUTS + i] <= '0;
                end
            end
        end

        if (mac_valid_out_r) begin
            for (int o = 0; o < OUTPUTS; o++) begin
                if (out_tile * OUTPUTS + o < OUTPUT_SIZE) begin
                    acc_buf[out_tile * OUTPUTS + o] <=
                        acc_buf[out_tile * OUTPUTS + o] + mac_outputs[o];
                end
            end
        end

        if (postproc_en) begin
            if (out_tile * OUTPUTS + pp_cnt < OUTPUT_SIZE) begin
                act_out_buf[out_tile * OUTPUTS + pp_cnt] <= activated_val;
            end
        end
    end
end

always_ff @(posedge clk) begin
    if (act_rd_addr < OUTPUT_SIZE) begin
        act_rd_data <= act_out_buf[act_rd_addr];
    end
    else begin
        act_rd_data <= '0;
    end
end

endmodule

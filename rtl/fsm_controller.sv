module fsm_controller #(
    parameter int INPUT_SIZE  = 784,
    parameter int OUTPUT_SIZE = 128,
    parameter int INPUTS      = 8,
    parameter int OUTPUTS     = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic mac_valid_out_r,
    output logic done,
    output logic fetch_en,
    output logic postproc_en,
    output logic [$clog2((INPUT_SIZE / INPUTS) + 1)-1:0]  in_tile,
    output logic [$clog2((OUTPUT_SIZE / OUTPUTS) + 1)-1:0] out_tile,
    output logic [$clog2(OUTPUTS + 1)-1:0]                pp_cnt
);

localparam int IN_TILES  = INPUT_SIZE / INPUTS;
localparam int OUT_TILES = OUTPUT_SIZE / OUTPUTS;

typedef enum logic [2:0] {
    IDLE,
    FETCH,
    DRAIN,
    POSTPROC,
    DONE_ST
} state_t;

state_t state;

assign fetch_en    = (state == FETCH);
assign postproc_en = (state == POSTPROC);
assign done        = (state == DONE_ST);

always_ff @(posedge clk) begin
    if (rst) begin
        state    <= IDLE;
        in_tile  <= '0;
        out_tile <= '0;
        pp_cnt   <= '0;
    end
    else begin
        case (state)
            IDLE: begin
                if (start) begin
                    in_tile  <= '0;
                    out_tile <= '0;
                    pp_cnt   <= '0;
                    state    <= FETCH;
                end
            end

            FETCH: begin
                state <= DRAIN;
            end

            DRAIN: begin
                if (mac_valid_out_r) begin
                    if (in_tile == IN_TILES - 1) begin
                        in_tile <= '0;
                        pp_cnt  <= '0;
                        state   <= POSTPROC;
                    end
                    else begin
                        in_tile <= in_tile + 1'b1;
                        state   <= FETCH;
                    end
                end
            end

            POSTPROC: begin
                if (pp_cnt == OUTPUTS - 1) begin
                    if (out_tile == OUT_TILES - 1) begin
                        state <= DONE_ST;
                    end
                    else begin
                        out_tile <= out_tile + 1'b1;
                        in_tile  <= '0;
                        pp_cnt   <= '0;
                        state    <= FETCH;
                    end
                end
                else begin
                    pp_cnt <= pp_cnt + 1'b1;
                end
            end

            DONE_ST: begin
                state <= IDLE;
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule

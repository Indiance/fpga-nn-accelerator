module fsm_controller #(
    parameter int INPUT_SIZE  = 784,
    parameter int OUTPUT_SIZE = 128,
    parameter int INPUTS      = 8,
    parameter int OUTPUTS     = 4,
    parameter int PIPELINE_LATENCY = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    output logic done,
    output logic fetch_en,
    output logic postproc_en,

    output logic [$clog2((INPUT_SIZE/INPUTS)+1)-1:0] in_tile,
    output logic [$clog2((OUTPUT_SIZE/OUTPUTS)+1)-1:0] out_tile,
    output logic [$clog2(OUTPUTS+1)-1:0] pp_cnt
);

localparam int IN_TILES  = INPUT_SIZE/INPUTS;
localparam int OUT_TILES = OUTPUT_SIZE/OUTPUTS;

typedef enum logic [2:0] {
    IDLE,
    STREAM,
    FLUSH,
    POSTPROC,
    DONE_ST
} state_t;

state_t state;

logic [$clog2(PIPELINE_LATENCY+1)-1:0] flush_cnt;

assign fetch_en    = (state == STREAM);
assign postproc_en = (state == POSTPROC);
assign done        = (state == DONE_ST);

always_ff @(posedge clk) begin

    if(rst) begin
        state     <= IDLE;
        in_tile   <= 0;
        out_tile  <= 0;
        pp_cnt    <= 0;
        flush_cnt <= 0;
    end

    else begin

        case(state)

        IDLE:
        begin
            if(start) begin
                in_tile   <= 0;
                out_tile  <= 0;
                pp_cnt    <= 0;
                flush_cnt <= 0;
                state     <= STREAM;
            end
        end

        STREAM:
        begin
            if(in_tile == IN_TILES-1) begin
                in_tile   <= 0;
                flush_cnt <= 0;
                state     <= FLUSH;
            end
            else begin
                in_tile <= in_tile + 1'b1;
            end
        end

        FLUSH:
        begin
            if(flush_cnt == PIPELINE_LATENCY-1) begin
                pp_cnt <= 0;
                state  <= POSTPROC;
            end
            else begin
                flush_cnt <= flush_cnt + 1'b1;
            end
        end

        POSTPROC:
        begin
            if(pp_cnt == OUTPUTS-1) begin
                if(out_tile == OUT_TILES-1) begin
                    state <= DONE_ST;
                end
                else begin
                    out_tile  <= out_tile + 1'b1;
                    in_tile   <= 0;
                    pp_cnt    <= 0;
                    flush_cnt <= 0;
                    state     <= STREAM;
                end
            end
            else begin
                pp_cnt <= pp_cnt + 1'b1;
            end
        end

        DONE_ST:
        begin
            state <= IDLE;
        end
        default:
            state <= IDLE;
        endcase
    end
end
endmodule

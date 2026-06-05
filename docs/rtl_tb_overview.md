# RTL and Testbench Overview

This directory pair contains the first SystemVerilog building blocks for the
FPGA neural-network accelerator: a fixed-point multiply-accumulate unit, a
parameterized MAC array, an activation buffer, and focused unit testbenches.

The project design document, `Complete Modelsim Revised Fpga Nn Project
Document.pdf`, frames these modules as part of a simulation-driven
ModelSim/Questa RTL implementation for fixed-point MNIST inference. It
identifies `mac_unit.sv` as the single multiply-accumulate pipeline and
`mac_array.sv` as the parameterized `M' x C'` parallel MAC structure used by
fully connected layer experiments.

## Files

- `rtl/mac_unit.sv`: parameterized signed fixed-point MAC stage.
- `rtl/mac_array.sv`: parameterized signed fixed-point dot-product pipeline.
- `rtl/act_buffer.sv`: parameterized activation storage with one write port
  and a multi-word registered read window.
- `tb/mac_unit_tb.sv`: unit test for positive, negative, accumulated, and zero
  Q8.8 multiply-accumulate cases.
- `tb/mac_array_tb.sv`: unit test for a 2-input by 2-output Q8.8 MAC array.

## Fixed-Point Convention

Both MAC modules use the project's default 16-bit signed Q8.8 convention:

- `DATA_WIDTH = 16`
- `FRAC_BITS = 8`
- one integer value represents `1 / 256`

The design document calls out Q8.8 fixed-point arithmetic as the default RTL
numeric format, with wider accumulation registers to reduce overflow risk.
That matches these files: operands are 16-bit signed values, while accumulated
results are 32-bit signed values.

## `mac_unit.sv`

`mac_unit` computes:

```text
acc_out = acc_in + ((a * b) >> FRAC_BITS)
```

With the default parameters, `a` and `b` are interpreted as signed Q8.8-style
fixed-point values. For example:

- `16'sd384` represents `1.5`
- `16'sd512` represents `2.0`
- Their scaled product is `16'sd768`, representing `3.0`

The multiplication result is held in a 32-bit signed intermediate. Because
multiplying two fixed-point values doubles the number of fractional bits, the
result is shifted right by `FRAC_BITS` before being added to `acc_in`.

### Interface Timing

`mac_unit` is a registered stage:

- `rst` is synchronous and clears `acc_out` and `valid_out`.
- `valid_in` is sampled on the rising edge of `clk`.
- If `valid_in` is high, `acc_out` updates on that clock edge.
- `valid_out` is the registered, one-cycle-delayed version of `valid_in`.
- If `valid_in` is low, `acc_out` holds its previous value.

## `mac_array.sv`

`mac_array` generalizes the single MAC operation into a parallel set of
dot-product pipelines:

```text
outputs[i] = sum((activations[j] * weights[i][j]) >> FRAC_BITS)
```

The parameters control the array shape:

- `INPUTS`: number of activation values consumed by each output dot product.
- `OUTPUTS`: number of output dot products computed in parallel.

In the design-document terminology, `INPUTS` corresponds to the input
parallelism `C'`, and `OUTPUTS` corresponds to output-neuron parallelism `M'`.
The estimated DSP cost for a fully parallel array scales with `M' x C'`, so
changing these parameters is the basic latency/resource tradeoff knob for fully
connected layers.

The current implementation builds each dot product from a chain of `mac_unit`
instances. For each output `o`, `INPUTS` MAC stages are connected through
`acc_chain[o][0]` through `acc_chain[o][INPUTS]`. The first accumulator value is
hard-wired to zero, and each stage adds one activation-weight product to the
running sum.

Because `mac_unit` is a registered stage, the array has pipeline latency. A
single valid input reaches the end of the accumulator chain after `INPUTS`
stages. The current implementation pipelines the valid signal through
`valid_pipe`, but it does not pipeline or latch the corresponding activation and
weight operands per stage. A controller using this version should keep the
`activations` and `weights` tile stable while the pipeline is consuming it, or a
future revision should add operand registers so each stage receives the matching
activation and weight for that dot-product transaction.

### Interface Timing

`mac_array` follows the same synchronous reset style as `mac_unit`, with a
longer pipeline:

- `rst` is synchronous and clears `outputs` and `valid_out`.
- `valid_in` is sampled on the rising edge of `clk`.
- If `valid_in` is high, a new dot-product transaction enters the MAC chain.
- `valid_pipe` tracks progress through the `INPUTS` MAC stages.
- `valid_out` is driven directly from `valid_pipe[INPUTS]`.
- In the current RTL, the final accumulator value reaches
  `acc_chain[k][INPUTS]` when `valid_out` is asserted, while the registered
  `outputs[k]` port captures that value on the following clock edge. A later
  cleanup should align `valid_out` with the registered `outputs` port if
  downstream modules consume `outputs` only when `valid_out` is high.
- If `valid_in` is low, the previous `outputs` values are retained.

### Array Data Layout

The ports use unpacked SystemVerilog arrays:

```systemverilog
input  logic signed [DATA_WIDTH-1:0] activations [INPUTS],
input  logic signed [DATA_WIDTH-1:0] weights [OUTPUTS][INPUTS],
output logic signed [31:0] outputs [OUTPUTS]
```

For each output index `i`, `weights[i][j]` is multiplied by
`activations[j]`. This is the weight-stationary interpretation described in
the project document: weights are treated as locally available while
activations are applied across the MAC array.

## `act_buffer.sv`

`act_buffer` stores activation values in a parameterized register-memory array
and returns a contiguous read window for the MAC array:

```systemverilog
parameter int DATA_WIDTH = 32,
parameter int DEPTH      = 128,
parameter int READ_WIDTH = 8
```

The write side stores one value per clock when `write_en` is high:

```systemverilog
mem[write_addr] <= write_data;
```

The read side returns `READ_WIDTH` consecutive values when `read_en` is high:

```systemverilog
activations[i] <= mem[read_addr + i];
```

This makes `read_addr` the base address of the activation window. For example,
with `READ_WIDTH = 8`, a read at address `16` returns memory entries `16`
through `23`.

### Interface Timing

- `rst` is synchronous and clears the registered `activations` outputs.
- Writes occur on the rising edge of `clk` when `write_en` is high.
- Reads occur on the rising edge of `clk` when `read_en` is high.
- The `activations` outputs are registered and retain their previous values
  when `read_en` is low.
- The module does not currently check that `read_addr + READ_WIDTH - 1` stays
  inside `DEPTH`; the controller or testbench should avoid out-of-range reads.
- If a write and read target the same address in one clock cycle, behavior
  depends on simulator scheduling and target memory inference style. Avoid
  same-cycle read/write address collisions unless the desired behavior is
  explicitly defined later.

## Testbench Behavior

`tb/mac_unit_tb.sv` generates a 100 MHz clock with `always #5 clk = ~clk`,
applies reset for 20 ns, then drives several single-MAC cases.

The first case is:

```text
a        = 384  // 1.5 in Q8.8
b        = 512  // 2.0 in Q8.8
acc_in   = 0
valid_in = 1
```

After one clock cycle, the expected `acc_out` value is `768`, which
corresponds to `3.0` in Q8.8 format. Additional cases check a negative
product, accumulation with a nonzero `acc_in`, and multiplication by zero.

`tb/mac_array_tb.sv` instantiates a smaller 2-input by 2-output array:

```text
activations = [1.0, 2.0]
weights     = [[1.0, 2.0],
               [3.0, 4.0]]
```

The expected Q8.8 outputs are:

- `outputs[0] = 5.0`, represented as `1280`
- `outputs[1] = 11.0`, represented as `2816`

The testbench timing should account for the pipelined `mac_array`
implementation. The original single-cycle array could be checked shortly after
`valid_in` was pulsed. The chained `mac_unit` version needs multiple clock
edges for the valid signal and accumulator values to move through the pipeline,
and the current registered `outputs` update one clock after `valid_out`.

## Example Simulation

With Icarus Verilog:

```sh
iverilog -g2012 -o mac_unit_tb.vvp rtl/mac_unit.sv tb/mac_unit_tb.sv
vvp mac_unit_tb.vvp

iverilog -g2012 -o mac_array_tb.vvp rtl/mac_array.sv tb/mac_array_tb.sv
vvp mac_array_tb.vvp
```

Expected `mac_unit` output:

```text
PASS: TEST1
PASS: TEST2
PASS: TEST3
PASS: TEST4
ALL TESTS COMPLETE
```

Expected `mac_array` output:

```text
PASS output0
PASS output1
ALL TESTS PASSED
```

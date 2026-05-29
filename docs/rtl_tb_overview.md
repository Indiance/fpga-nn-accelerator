# RTL and Testbench Overview

This directory pair contains the first SystemVerilog building block for the FPGA neural-network accelerator: a fixed-point multiply-accumulate unit and a minimal testbench.

## Files

- `rtl/mac_unit.sv`: parameterized signed fixed-point MAC stage.
- `tb/mac_unit_tb.sv`: smoke test for one Q8.8 multiply-accumulate operation.

## `mac_unit`

`mac_unit` computes:

```text
acc_out = acc_in + ((a * b) >> FRAC_BITS)
```

The default parameters are:

- `DATA_WIDTH = 16`
- `FRAC_BITS = 8`

With these defaults, `a` and `b` are interpreted as signed Q8.8-style fixed-point values. For example:

- `16'sd384` represents `1.5`
- `16'sd512` represents `2.0`
- Their scaled product is `16'sd768`, representing `3.0`

The multiplication result is held in a 32-bit signed intermediate. Because multiplying two fixed-point values doubles the number of fractional bits, the result is shifted right by `FRAC_BITS` before being added to `acc_in`.

## Interface Timing

`mac_unit` is a registered stage:

- `rst` is synchronous and clears `acc_out` and `valid_out`.
- `valid_in` is sampled on the rising edge of `clk`.
- If `valid_in` is high, `acc_out` updates on that clock edge.
- `valid_out` is the registered, one-cycle-delayed version of `valid_in`.
- If `valid_in` is low, `acc_out` holds its previous value.

## Testbench Behavior

`tb/mac_unit_tb.sv` generates a 100 MHz clock with `always #5 clk = ~clk`, applies reset for 20 ns, then drives:

```text
a        = 384  // 1.5 in Q8.8
b        = 512  // 2.0 in Q8.8
acc_in   = 0
valid_in = 1
```

After one clock cycle, the expected `acc_out` value is `768`, which corresponds to `3.0` in Q8.8 format.

## Example Simulation

With Icarus Verilog:

```sh
iverilog -g2012 -o mac_unit_tb.vvp rtl/mac_unit.sv tb/mac_unit_tb.sv
vvp mac_unit_tb.vvp
```

Expected output:

```text
acc_out =         768
```

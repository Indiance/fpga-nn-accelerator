# MNIST MLP Model Training, Quantization & Tiling Guide

This document describes the design, quantization, tiling, and verification of the 3-Layer Multilayer Perceptron (MLP) MNIST model implemented in [fixed_point_mlp_infer.py](file:///home/user/Desktop/fpga-nn-accelerator/python/fixed_point_mlp_infer.py).

---

## 1. Network Architecture
The hardware accelerator is designed specifically for a feedforward MLP (Fully Connected) architecture. The model is structured as follows:

```
  Input Image (28x28)
         │
         ▼
    [ Flatten ] ────► 784 Inputs
         │
         ▼
      [ FC1 ]   ────► 784 -> 128 (ReLU Enabled)
         │
         ▼
      [ FC2 ]   ────► 128 -> 64 (ReLU Enabled)
         │
         ▼
      [ FC3 ]   ────► 64 -> 10  (ReLU Disabled / Raw Scores)
         │
         ▼
    [ Argmax ]  ────► Predicted Class (0-9)
```

### Layer Parameters & Parallelism
The layers use different structural tiling dimensions ($M'$ output parallelism, $C'$ input parallelism) matching the hardware configuration:

| Layer | Input Size ($C$) | Output Size ($M$) | Hardware Input Parallelism ($C'$) | Hardware Output Parallelism ($M'$) | MACs/Cycle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FC1** | 784 | 128 | 8 | 4 | 32 |
| **FC2** | 128 | 64 | 8 | 4 | 32 |
| **FC3** | 64 | 10 | 8 | 2 | 16 |

---

## 2. Fixed-Point Quantization (Q8.8)
To run inference efficiently in hardware without floating-point units, weights, biases, and activations are quantized to the **Q8.8 signed fixed-point format** (16-bit total width, 8 fractional bits):

$$\text{Quantized Value} = \text{round}(\text{Float Value} \times 256)$$

### Saturation Guarding
To prevent overflow and underflow wrapping during fixed-point addition or scaling, values are strictly clipped to signed 16-bit integer boundaries:
* **Max Limit**: `32767` (`0x7FFF`)
* **Min Limit**: `-32768` (`0x8000`)

### Bias Addition and Activation (ReLU)
Post-processing for each fully connected neuron is performed combinational-wise:
1. **Bias Accumulation**: Sign-extends and adds the 16-bit bias to the 32-bit accumulated sum.
2. **ReLU Activation**: If ReLU is enabled for the layer, any negative values are clamped to `0`.
3. **16-bit Clipping**: Saturation logic clips the result back to signed 16-bit limits before forwarding to the next layer.

---

## 3. Custom Weight Tiling Layout
The hardware fully connected engine [layer_fc.sv](file:///home/user/Desktop/fpga-nn-accelerator/rtl/layer_fc.sv) does not consume weight matrices in standard row-major format. Instead, weights are tiled in segments of size $M' \times C'$ to match the physical MAC array:

```
  Row-Major Matrix            Hardware-Tiled Buffer
┌───┬───┬───┬───┐          ┌──────────────┐
│ 0 │ 1 │ 2 │ 3 │          │ Tile (0,0)   │ ◄── (outputs=M', inputs=C')
├───┼───┼───┼───┤   ───►   ├──────────────┤
│ 4 │ 5 │ 6 │ 7 │          │ Tile (0,1)   │
└───┴───┴───┴───┘          └──────────────┘
```

The script `fixed_point_mlp_infer.py` automatically handles this tiling transformation using the function `export_tiled_weights()` and generates:
* `weights/fc1_weights.mem` (tiled $4 \times 8$)
* `weights/fc2_weights.mem` (tiled $4 \times 8$)
* `weights/fc3_weights.mem` (tiled $2 \times 8$)

Biases are exported linearly to `weights/fc[1/2/3]_bias.mem` (one 16-bit hex word per line).

---

## 4. Bit-Accurate Verification Flow
To ensure that the SystemVerilog simulation matches the Python reference model perfectly, `fixed_point_mlp_infer.py` implements a **bit-accurate fixed-point emulator** in Python:

1. **Quantization**: Normalizes and quantizes 100 actual test images from the MNIST dataset.
2. **Emulated Inference**: Runs the quantized images through the layers using bitwise shifts (`>> 8`) and integer bounds checks identical to the hardware math.
3. **Reference Output**: Computes the expected prediction class for each image.
4. **Test Vector Export**: Writes the inputs to `tb_input_images.mem` and expectations to `tb_expected_classes.mem`.

This matching verification flow ensures that when the SystemVerilog testbench compiles and runs, it yields a **100% classification match** against the reference Python model.

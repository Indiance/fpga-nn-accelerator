# MNIST Model Performance and Weight Quantization

## PyTorch Model Summary

The PyTorch reference model is defined in `python/mnist_model.py` as a compact
CNN for 28x28 grayscale MNIST images. It uses two convolution blocks followed by
two fully connected layers:

- `Conv2d(1, 16, kernel_size=3, padding=1)`
- ReLU and 2x2 max pooling
- `Conv2d(16, 32, kernel_size=3, padding=1)`
- ReLU and 2x2 max pooling
- `Linear(32 * 7 * 7, 128)`
- ReLU
- `Linear(128, 10)`

Training is handled by `python/train_mnist.py` with normalized MNIST inputs,
cross-entropy loss, and Adam. The current saved `best_mnist_cnn.pt` checkpoint
was produced after 5 epochs and records a test accuracy of 0.9892, or 98.92%.
For this project, that level of accuracy is high enough to make the PyTorch
model a useful functional reference for the FPGA implementation.

## Compute and Parameter Cost

The model is intentionally small, but it is still dominated by multiply-
accumulate work:

| Layer | Learned values | Approximate MACs per image |
| --- | ---: | ---: |
| `features.0.weight` + bias | 160 | 112,896 |
| `features.3.weight` + bias | 4,640 | 903,168 |
| `classifier.1.weight` + bias | 200,832 | 200,704 |
| `classifier.3.weight` + bias | 1,290 | 1,280 |
| **Total** | **206,922** | **1,218,048** |

The first fully connected layer contains most of the stored parameters, while
the second convolution layer contributes most of the arithmetic. This split is
important for the accelerator design: memory layout and bandwidth matter for
the classifier weights, while convolution throughput matters for inference
latency.

## Q8.8 Export Status

The export flow in `python/export_questa_mem_q8_8.py` converts trained PyTorch
tensors to signed INT16 Q8.8 values using:

```text
quantized = round(weight * 256)
```

The generated manifests under `weights/` show:

- total quantized words: 206,922
- storage as FP32 weights: 827,688 bytes
- storage as INT16 Q8.8 weights: 413,844 bytes
- observed saturation count: 0

No saturation means every trained weight and bias in the current checkpoint fits
inside the Q8.8 numeric range. The largest observed source range is about
-0.7126 to 0.5183, which is well within the representable Q8.8 range of roughly
-128 to 127.996 with a step size of 1/256.

## Why Quantizing Weights Matters

PyTorch training uses floating-point tensors because floating point is flexible,
well supported on CPUs and GPUs, and convenient for gradient-based learning.
The FPGA inference path has different constraints. A hardware accelerator is
usually built around fixed-width integer datapaths, predictable memory layouts,
and deterministic arithmetic. Quantizing the weights bridges that gap.

Quantized weights are important for several reasons:

- **Hardware feasibility:** Fixed-point multipliers, adders, and accumulators
  map more directly to FPGA DSP blocks and LUT fabric than full floating-point
  units.
- **Memory reduction:** INT16 Q8.8 cuts weight storage in half compared with
  FP32. For this model, the learned parameters drop from 827,688 bytes to
  413,844 bytes.
- **Bandwidth reduction:** Smaller weights reduce the amount of data fetched
  from BRAM or external memory during inference, which is especially useful for
  the large `classifier.1.weight` tensor.
- **Simulation compatibility:** The `.mem` files are emitted as 16-bit hex
  words suitable for `$readmemh`, so the same exported data can be loaded by
  Questa or ModelSim testbenches.
- **Deterministic inference:** Fixed-point arithmetic makes it easier to
  reproduce hardware behavior across simulation and synthesis.

Quantization is not free. Rounding changes model values, and too few integer or
fractional bits can reduce accuracy or cause overflow. The current Q8.8 export
is a reasonable fit for this checkpoint because the trained weights are small,
the quantization step is fine enough for MNIST, and the manifests report zero
saturated values. Hardware validation should still compare FPGA or RTL outputs
against the PyTorch reference on a representative MNIST test set.

## Practical Interpretation

The PyTorch model establishes the expected classification behavior: about 98.92%
test accuracy for the saved checkpoint. The Q8.8 exported weights establish the
hardware-facing representation of that same trained model. Keeping these two
views connected is critical: PyTorch is the accuracy reference, while the
quantized `.mem` files are the data that the FPGA implementation can actually
consume.

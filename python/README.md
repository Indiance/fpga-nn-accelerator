# MNIST PyTorch Training and Weight Export

This directory contains the PyTorch training flow for MNIST. Simulator-facing
weight images are exported outside this package under `sim/questa/`.

## Setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_mnist.py --epochs 5 --batch-size 128
```

By default the script:

- downloads MNIST into `python/data/`
- saves checkpoints into `python/checkpoints/`
- exports learned tensors into `python/exports/mnist_cnn_weights.npz`

## Export Q8.8 Questa Memory Files

```bash
python export_questa_mem_q8_8.py
```

This quantizes the `.npz` exports and `.pt` checkpoints to signed INT16 Q8.8
(`round(weight * 256)`) and writes Questa-compatible `$readmemh` files under
`../sim/questa/mem/q8_8/`.

Each source gets its own directory with:

- one 16-bit hex word-per-line `.mem` file per tensor for `$readmemh`
- `all_weights.mem` with tensors concatenated in state-dict order
- `manifest.json` with shapes, word offsets, quantized ranges, and saturation counts

## Performance and Quantization Notes

See `../docs/mnist_performance_quantization.md` for the current PyTorch MNIST
checkpoint performance, model compute/parameter breakdown, and why the trained
weights are quantized before being used by the FPGA simulation flow.

## Useful Options

```bash
python train_mnist.py --epochs 10 --lr 0.001 --device cuda
python train_mnist.py --epochs 3 --no-export
python train_mnist.py --checkpoint checkpoints/best_mnist_cnn.pt
python export_questa_mem_q8_8.py checkpoints/best_mnist_cnn.pt
python export_questa_mem_q8_8.py --endianness little-bytes --write-numpy
```

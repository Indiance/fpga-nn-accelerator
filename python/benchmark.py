import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Configs matching fixed_point_mlp_infer.py
FRAC_BITS = 8
SCALE = 1 << FRAC_BITS
INT16_MIN = -32768
INT16_MAX = 32767

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def quantize_and_clip(val):
    scaled = np.rint(val * SCALE)
    clipped = np.clip(scaled, INT16_MIN, INT16_MAX)
    return clipped.astype(np.int32)

def fixed_point_infer(x, W1, b1, W2, b2, W3, b3):
    # Layer 1
    acc1 = []
    for o in range(128):
        sum_val = 0
        for i in range(784):
            mult = int(x[i]) * int(W1[o, i])
            sum_val += (mult >> FRAC_BITS)
        biased = sum_val + int(b1[o])
        activated = np.clip(max(0, biased), INT16_MIN, INT16_MAX)
        acc1.append(activated)
    acc1 = np.array(acc1, dtype=np.int32)

    # Layer 2
    acc2 = []
    for o in range(64):
        sum_val = 0
        for i in range(128):
            mult = int(acc1[i]) * int(W2[o, i])
            sum_val += (mult >> FRAC_BITS)
        biased = sum_val + int(b2[o])
        activated = np.clip(max(0, biased), INT16_MIN, INT16_MAX)
        acc2.append(activated)
    acc2 = np.array(acc2, dtype=np.int32)

    # Layer 3
    acc3 = []
    for o in range(10):
        sum_val = 0
        for i in range(64):
            mult = int(acc2[i]) * int(W3[o, i])
            sum_val += (mult >> FRAC_BITS)
        biased = sum_val + int(b3[o])
        # No ReLU on layer 3, only saturation
        activated = np.clip(biased, INT16_MIN, INT16_MAX)
        acc3.append(activated)
    acc3 = np.array(acc3, dtype=np.int32)

    return acc3

def main():
    print("==================================================")
    print("     MNIST MLP Latency & Performance Benchmark    ")
    print("==================================================")

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_path = repo_root / "python" / "checkpoints" / "best_mlp.pt"

    # Initialize model
    model = MNISTMLP()
    
    # Load trained weights if they exist, otherwise initialize randomly
    weights_loaded = False
    if checkpoint_path.exists():
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"[Info] Loaded trained model weights from {checkpoint_path.name}")
            weights_loaded = True
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}. Using random weights.")
    else:
        print("[Info] No trained checkpoint found. Using randomly initialized weights.")
        print("       (Run 'python python/fixed_point_mlp_infer.py' to train first.)")

    model.eval()

    # Extract weights for the Python Fixed-Point Emulator
    state_dict = model.state_dict()
    W1 = state_dict['fc1.weight'].numpy()
    b1 = state_dict['fc1.bias'].numpy()
    W2 = state_dict['fc2.weight'].numpy()
    b2 = state_dict['fc2.bias'].numpy()
    W3 = state_dict['fc3.weight'].numpy()
    b3 = state_dict['fc3.bias'].numpy()

    W1_q = quantize_and_clip(W1)
    b1_q = quantize_and_clip(b1)
    W2_q = quantize_and_clip(W2)
    b2_q = quantize_and_clip(b2)
    W3_q = quantize_and_clip(W3)
    b3_q = quantize_and_clip(b3)

    # Create dummy single-image input
    dummy_input_torch = torch.randn(1, 784)
    dummy_input_np = dummy_input_torch.numpy().flatten()
    dummy_input_q = quantize_and_clip(dummy_input_np)

    # 1. Benchmark CPU PyTorch FP32 Inference (Batch Size = 1)
    print("\nBenchmarking PyTorch CPU (FP32, Batch Size = 1)...")
    with torch.no_grad():
        # Warm-up
        for _ in range(100):
            _ = model(dummy_input_torch)
        
        # Timing
        start_time = time.perf_counter()
        iters = 2000
        for _ in range(iters):
            _ = model(dummy_input_torch)
        end_time = time.perf_counter()
        
    cpu_latency_bs1 = ((end_time - start_time) / iters) * 1e6  # in microseconds
    print(f"-> Average Latency: {cpu_latency_bs1:.2f} μs")

    # 2. Benchmark MPS PyTorch FP32 Inference (Batch Size = 1) on Mac
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        print("\nBenchmarking PyTorch MPS (Metal GPU, FP32, Batch Size = 1)...")
        model.to("mps")
        dummy_input_mps = dummy_input_torch.to("mps")
        with torch.no_grad():
            # Warm-up
            for _ in range(100):
                _ = model(dummy_input_mps)
            torch.mps.synchronize()
            
            # Timing
            start_time = time.perf_counter()
            iters_mps = 2000
            for _ in range(iters_mps):
                _ = model(dummy_input_mps)
            torch.mps.synchronize()
            end_time = time.perf_counter()
        
        mps_latency_bs1 = ((end_time - start_time) / iters_mps) * 1e6  # in microseconds
        print(f"-> Average Latency: {mps_latency_bs1:.2f} μs")
    else:
        print("\n[MPS GPU not available or not supported by PyTorch on this environment]")
        mps_latency_bs1 = None

    # 3. Benchmark Python Fixed-Point Emulator (Batch Size = 1)
    print("\nBenchmarking Python Fixed-Point Emulator (Batch Size = 1)...")
    # Warm-up
    for _ in range(10):
        _ = fixed_point_infer(dummy_input_q, W1_q, b1_q, W2_q, b2_q, W3_q, b3_q)
    
    # Timing (Python loop is slower, so we use fewer iterations)
    start_time = time.perf_counter()
    iters_emu = 200
    for _ in range(iters_emu):
        _ = fixed_point_infer(dummy_input_q, W1_q, b1_q, W2_q, b2_q, W3_q, b3_q)
    end_time = time.perf_counter()
    
    emu_latency = ((end_time - start_time) / iters_emu) * 1e6  # in microseconds
    print(f"-> Average Latency: {emu_latency:.2f} μs (note: Python interpreter overhead)")

    # 4. Throughput Benchmarks (Batch Size = 128)
    print("\nBenchmarking Throughput (Batch Size = 128)...")
    batch_input_torch = torch.randn(128, 784)
    with torch.no_grad():
        # CPU
        model.to("cpu")
        start_time = time.perf_counter()
        for _ in range(100):
            _ = model(batch_input_torch)
        cpu_tp_time = time.perf_counter() - start_time
        cpu_throughput = (128 * 100) / cpu_tp_time
        
        # MPS
        if mps_available:
            model.to("mps")
            batch_input_mps = batch_input_torch.to("mps")
            start_time = time.perf_counter()
            for _ in range(100):
                _ = model(batch_input_mps)
            torch.mps.synchronize()
            mps_tp_time = time.perf_counter() - start_time
            mps_throughput = (128 * 100) / mps_tp_time
        else:
            mps_throughput = None

    # 5. FPGA Verilog Model Latency (Analytically Calculated for Config A)
    # Total Cycles = 20,434 cycles
    fpga_cycles = 20434
    fpga_100mhz_latency = (fpga_cycles / 100e6) * 1e6  # 204.34 μs
    fpga_200mhz_latency = (fpga_cycles / 200e6) * 1e6  # 102.17 μs
    fpga_50mhz_latency  = (fpga_cycles / 50e6) * 1e6   # 408.68 μs

    # Print Results Table
    print("\n" + "="*70)
    print(f"{'Platform / Model':<38} | {'Latency (μs)':<14} | {'Throughput (img/sec)':<14}")
    print("-"*70)
    
    print(f"{'PyTorch CPU FP32 (BS=1)':<38} | {cpu_latency_bs1:>11.2f} μs | {1e6/cpu_latency_bs1:>17.1f}")
    if mps_available:
        print(f"{'PyTorch MPS GPU FP32 (BS=1)':<38} | {mps_latency_bs1:>11.2f} μs | {1e6/mps_latency_bs1:>17.1f}")
    else:
        print(f"{'PyTorch MPS GPU FP32 (BS=1)':<38} | {'N/A':>14} | {'N/A':>20}")
        
    print(f"{'Python Fixed-Point Emulator (BS=1)':<38} | {emu_latency:>11.2f} μs | {1e6/emu_latency:>17.1f}")
    print("-"*70)
    print(f"{'PyTorch CPU FP32 (BS=128 Batch)':<38} | {'--':>14} | {cpu_throughput:>20.1f}")
    if mps_available:
        print(f"{'PyTorch MPS GPU FP32 (BS=128 Batch)':<38} | {'--':>14} | {mps_throughput:>20.1f}")
    print("-"*70)
    print(f"{'Verilog FPGA Model @  50 MHz (BS=1)':<38} | {fpga_50mhz_latency:>11.2f} μs | {1e6/fpga_50mhz_latency:>17.1f}")
    print(f"{'Verilog FPGA Model @ 100 MHz (BS=1)':<38} | {fpga_100mhz_latency:>11.2f} μs | {1e6/fpga_100mhz_latency:>17.1f}")
    print(f"{'Verilog FPGA Model @ 200 MHz (BS=1)':<38} | {fpga_200mhz_latency:>11.2f} μs | {1e6/fpga_200mhz_latency:>17.1f}")
    print("="*70)
    
    print("\nNote on Latency and Throughput comparison:")
    print("1. PyTorch batch size = 1 latency is dominated by framework overhead, Python-to-C++ transitions, and API calls.")
    print("2. The Verilog FPGA model operates directly at the register-transfer level with zero framework overhead.")
    print("3. At batch size 128, PyTorch achieves massive parallelism, exploiting the parallel CPU vector instructions or GPU cores.")
    print("4. To run this benchmark on your trained weights, first train the model by running:")
    print("   python python/fixed_point_mlp_infer.py")
    print("==================================================")

if __name__ == "__main__":
    main()

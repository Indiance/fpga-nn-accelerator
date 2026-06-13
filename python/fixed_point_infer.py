import numpy as np
from pathlib import Path

def fixed_point_infer_layer(x, W, b, frac_bits=8):
    """
    Computes single fully-connected layer in fixed-point matching layer_fc.sv.
    
    x: shape (INPUT_SIZE,) - Q8.8 inputs
    W: shape (OUTPUT_SIZE, INPUT_SIZE) - Q8.8 weights
    b: shape (OUTPUT_SIZE,) - Q8.8 biases
    """
    scale = 1 << frac_bits
    
    # 16-bit signed integer limits for saturation
    INT16_MIN = -32768
    INT16_MAX = 32767
    
    output_size = W.shape[0]
    input_size = W.shape[1]
    
    # Simulating the accumulation per neuron
    outputs = []
    for o in range(output_size):
        acc = 0
        for i in range(input_size):
            # Multiply and shift immediately per MAC unit
            mult = int(x[i]) * int(W[o, i])
            # Arithmetic right shift
            shifted = mult >> frac_bits
            acc += shifted
            
        # Add bias (sign-extended bias added to accumulator)
        biased_val = acc + int(b[o])
        
        # Apply ReLU if enabled, then 16-bit saturation clipping
        activated = biased_val
        if activated < 0:
            activated = 0
        elif activated > INT16_MAX:
            activated = INT16_MAX
        elif activated < INT16_MIN:
            activated = INT16_MIN
            
        outputs.append(activated)
        
    return np.array(outputs, dtype=np.int32)

def to_hex_word(val):
    return f"{int(val) & 0xFFFF:04X}"

def main():
    np.random.seed(42)
    
    INPUT_SIZE = 16
    OUTPUT_SIZE = 8
    INPUTS = 4
    OUTPUTS = 2
    
    IN_TILES = INPUT_SIZE // INPUTS
    OUT_TILES = OUTPUT_SIZE // OUTPUTS
    
    # Generate random test values in range [-2.0, 2.0]
    x_float = np.random.uniform(-2.0, 2.0, size=(INPUT_SIZE,))
    W_float = np.random.uniform(-1.5, 1.5, size=(OUTPUT_SIZE, INPUT_SIZE))
    b_float = np.random.uniform(-1.0, 1.0, size=(OUTPUT_SIZE,))
    
    # Scale to Q8.8
    x_q = np.round(x_float * 256).astype(np.int32)
    W_q = np.round(W_float * 256).astype(np.int32)
    b_q = np.round(b_float * 256).astype(np.int32)
    
    # Run reference inference
    out_q = fixed_point_infer_layer(x_q, W_q, b_q, frac_bits=8)
    
    # Print out results for verification visibility
    print("Inputs (Q8.8):", x_q)
    print("Biases (Q8.8):", b_q)
    print("Expected Outputs (Q8.8):", out_q)
    
    # ----------------------------------------------------
    # Export files
    # ----------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    
    # Formats: one 16-bit hex word per line
    # Write inputs.mem
    inputs_hex = [to_hex_word(val) for val in x_q]
    
    # Tiled weight formatting:
    # We tile the OUTPUT_SIZE x INPUTS weight matrix in row-major tile format
    tiled_weights = []
    for out_tile in range(OUT_TILES):
        for in_tile in range(IN_TILES):
            for o in range(OUTPUTS):
                for i in range(INPUTS):
                    neuron_out = out_tile * OUTPUTS + o
                    neuron_in = in_tile * INPUTS + i
                    tiled_weights.append(W_q[neuron_out, neuron_in])
                    
    weights_hex = [to_hex_word(val) for val in tiled_weights]
    biases_hex = [to_hex_word(val) for val in b_q]
    outputs_hex = [to_hex_word(val) for val in out_q]
    
    # Write files in repo root and in tb/ directory so they are accessible from anywhere
    for dest_dir in [repo_root, repo_root / "tb"]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        with open(dest_dir / "tb_inputs.mem", "w") as f:
            f.write("\n".join(inputs_hex) + "\n")
            
        with open(dest_dir / "tb_weights.mem", "w") as f:
            f.write("\n".join(weights_hex) + "\n")
            
        with open(dest_dir / "tb_biases.mem", "w") as f:
            f.write("\n".join(biases_hex) + "\n")
            
        with open(dest_dir / "tb_expected_outputs.mem", "w") as f:
            f.write("\n".join(outputs_hex) + "\n")
            
    print("Test vectors written successfully to repo root and tb/ directory!")

if __name__ == "__main__":
    main()

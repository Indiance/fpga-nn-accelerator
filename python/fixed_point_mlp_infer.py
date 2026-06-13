import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configurations
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

def to_hex_word(val):
    return f"{int(val) & 0xFFFF:04X}"

def quantize_and_clip(val):
    scaled = np.rint(val * SCALE)
    clipped = np.clip(scaled, INT16_MIN, INT16_MAX)
    return clipped.astype(np.int32)

def export_tiled_weights(W_q, filename, outputs, inputs):
    # W_q shape: (OUTPUT_SIZE, INPUT_SIZE)
    output_size, input_size = W_q.shape
    in_tiles = input_size // inputs
    out_tiles = output_size // outputs
    
    tiled = []
    for out_tile in range(out_tiles):
        for in_tile in range(in_tiles):
            for o in range(outputs):
                for i in range(inputs):
                    neuron_out = out_tile * outputs + o
                    neuron_in = in_tile * inputs + i
                    tiled.append(W_q[neuron_out, neuron_in])
                    
    with open(filename, 'w') as f:
        for val in tiled:
            f.write(f"{to_hex_word(val)}\n")

def export_bias(b_q, filename):
    with open(filename, 'w') as f:
        for val in b_q:
            f.write(f"{to_hex_word(val)}\n")

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
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "python" / "data"
    weights_dir = repo_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = repo_root / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training 3-layer MLP on {device} for 3 epochs...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        print(f"Epoch {epoch+1} - Loss: {total_loss/total:.4f}, Accuracy: {correct/total*100:.2f}%")

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
            test_total += images.size(0)
    test_acc = test_correct / test_total * 100
    print(f"Float model test accuracy: {test_acc:.2f}%")

    # Save checkpoint
    torch.save(model.state_dict(), repo_root / "python" / "checkpoints" / "best_mlp.pt")

    # Quantize weights
    state_dict = model.state_dict()
    W1 = state_dict['fc1.weight'].cpu().numpy()
    b1 = state_dict['fc1.bias'].cpu().numpy()
    W2 = state_dict['fc2.weight'].cpu().numpy()
    b2 = state_dict['fc2.bias'].cpu().numpy()
    W3 = state_dict['fc3.weight'].cpu().numpy()
    b3 = state_dict['fc3.bias'].cpu().numpy()

    W1_q = quantize_and_clip(W1)
    b1_q = quantize_and_clip(b1)
    W2_q = quantize_and_clip(W2)
    b2_q = quantize_and_clip(b2)
    W3_q = quantize_and_clip(W3)
    b3_q = quantize_and_clip(b3)

    print("Exporting tiled weight/bias memory files...")
    # Tiling shapes matching layer_fc config:
    # FC1: INPUTS=8, OUTPUTS=4
    # FC2: INPUTS=8, OUTPUTS=4
    # FC3: INPUTS=8, OUTPUTS=2
    export_tiled_weights(W1_q, weights_dir / "fc1_weights.mem", outputs=4, inputs=8)
    export_bias(b1_q, weights_dir / "fc1_bias.mem")
    
    export_tiled_weights(W2_q, weights_dir / "fc2_weights.mem", outputs=4, inputs=8)
    export_bias(b2_q, weights_dir / "fc2_bias.mem")
    
    export_tiled_weights(W3_q, weights_dir / "fc3_weights.mem", outputs=2, inputs=8)
    export_bias(b3_q, weights_dir / "fc3_bias.mem")

    # Copy files to root and tb/ so simulator can find them
    for dest in [repo_root, tb_dir]:
        dest_weights_dir = dest / "weights"
        if weights_dir != dest_weights_dir:
            dest_weights_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(weights_dir / "fc1_weights.mem", dest_weights_dir / "fc1_weights.mem")
            shutil.copy(weights_dir / "fc1_bias.mem", dest_weights_dir / "fc1_bias.mem")
            shutil.copy(weights_dir / "fc2_weights.mem", dest_weights_dir / "fc2_weights.mem")
            shutil.copy(weights_dir / "fc2_bias.mem", dest_weights_dir / "fc2_bias.mem")
            shutil.copy(weights_dir / "fc3_weights.mem", dest_weights_dir / "fc3_weights.mem")
            shutil.copy(weights_dir / "fc3_bias.mem", dest_weights_dir / "fc3_bias.mem")

    # Generate 100 test images and reference classes
    print("Generating 100 bit-accurate test vectors...")
    input_mem_lines = []
    expected_classes = []

    test_loader_1 = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    fp_correct = 0
    count = 0
    for img, label in test_loader_1:
        if count >= 100:
            break
        img_np = img.squeeze().numpy().flatten()
        img_q = quantize_and_clip(img_np)
        
        # Run bit-accurate fixed point inference
        fp_out = fixed_point_infer(img_q, W1_q, b1_q, W2_q, b2_q, W3_q, b3_q)
        pred_class = np.argmax(fp_out)
        
        if pred_class == label.item():
            fp_correct += 1
            
        expected_classes.append(pred_class)
        
        # Append image inputs (784 lines per image)
        for val in img_q:
            input_mem_lines.append(to_hex_word(val))
            
        count += 1

    fp_acc = fp_correct / 100 * 100
    print(f"Fixed-point reference test accuracy (on first 100 images): {fp_acc:.2f}%")
    assert fp_acc >= 90.0, f"Error: Fixed-point accuracy is {fp_acc}%, which is less than the required 90%!"

    # Write test inputs and classes to file
    for dest_dir in [repo_root, tb_dir]:
        with open(dest_dir / "tb_input_images.mem", "w") as f:
            f.write("\n".join(input_mem_lines) + "\n")
        with open(dest_dir / "tb_expected_classes.mem", "w") as f:
            for c in expected_classes:
                f.write(f"{c:01X}\n") # write as 1-digit hex for easy reading

    print(f"All files exported successfully! Test image accuracy is {fp_acc}%.")

if __name__ == "__main__":
    main()

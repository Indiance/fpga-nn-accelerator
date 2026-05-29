from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS
INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize exported weights/checkpoints to signed INT16 Q8.8 Questa .mem files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input .npz or PyTorch .pt/.pth files. Defaults to python/exports/*.npz and python/checkpoints/*.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sim" / "questa" / "mem" / "q8_8",
        help="Directory for generated Questa-compatible memory files.",
    )
    parser.add_argument(
        "--endianness",
        choices=["word", "little-bytes", "big-bytes"],
        default="word",
        help=(
            "Hex line format. 'word' writes one 16-bit word per line for $readmemh; "
            "'little-bytes'/'big-bytes' write two byte tokens per value."
        ),
    )
    parser.add_argument(
        "--write-numpy",
        action="store_true",
        help="Also write quantized .npy/.npz files for Python-side inspection.",
    )
    return parser.parse_args()


def default_inputs() -> list[Path]:
    return sorted((ROOT_DIR / "exports").glob("*.npz")) + sorted(
        (ROOT_DIR / "checkpoints").glob("*.pt")
    )


def sanitize_name(name: str) -> str:
    return (
        name.replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {name: np.asarray(npz[name]) for name in npz.files}


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"PyTorch is required to read checkpoint input {path}."
        ) from exc

    checkpoint: Any = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        if hasattr(tensor, "detach"):
            arrays[name] = tensor.detach().cpu().numpy()
    return arrays


def load_weights(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix in {".pt", ".pth"}:
        return load_checkpoint(path)
    raise ValueError(f"Unsupported input file type: {path}")


def quantize_q8_8(array: np.ndarray) -> tuple[np.ndarray, int]:
    scaled = np.rint(array.astype(np.float64) * SCALE)
    clipped = np.clip(scaled, INT16_MIN, INT16_MAX)
    saturated_count = int(np.count_nonzero(scaled != clipped))
    return clipped.astype(np.int16), saturated_count


def int16_to_hex_word(value: np.int16) -> str:
    return f"{int(value) & 0xFFFF:04X}"


def int16_to_hex_tokens(value: np.int16, endianness: str) -> list[str]:
    word = int(value) & 0xFFFF
    high = (word >> 8) & 0xFF
    low = word & 0xFF
    if endianness == "little-bytes":
        return [f"{low:02X}", f"{high:02X}"]
    if endianness == "big-bytes":
        return [f"{high:02X}", f"{low:02X}"]
    return [f"{word:04X}"]


def write_mem(path: Path, values: np.ndarray, endianness: str) -> None:
    with path.open("w", encoding="ascii") as f:
        for value in values.reshape(-1):
            f.write("\n".join(int16_to_hex_tokens(value, endianness)))
            f.write("\n")


def tensor_manifest(
    name: str,
    array: np.ndarray,
    quantized: np.ndarray,
    saturated_count: int,
    mem_path: Path,
    base_word_address: int,
) -> dict[str, Any]:
    flat = quantized.reshape(-1)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "word_count": int(flat.size),
        "base_word_address": base_word_address,
        "source_min": float(np.min(array)),
        "source_max": float(np.max(array)),
        "q8_8_min": int(np.min(flat)),
        "q8_8_max": int(np.max(flat)),
        "saturated_count": saturated_count,
        "mem_file": mem_path.name,
    }


def export_input(
    path: Path,
    output_dir: Path,
    endianness: str,
    write_numpy: bool,
) -> dict[str, Any]:
    arrays = load_weights(path)
    target_dir = output_dir / sanitize_name(path.stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": str(path),
        "format": "signed INT16 Q8.8",
        "fractional_bits": FRACTIONAL_BITS,
        "scale": SCALE,
        "endianness": endianness,
        "flatten_order": "C row-major",
        "simulator": "Intel Questa / ModelSim",
        "tensors": [],
    }

    all_values: list[np.ndarray] = []
    base_word_address = 0
    for name, array in arrays.items():
        quantized, saturated_count = quantize_q8_8(array)
        safe_name = sanitize_name(name)
        mem_path = target_dir / f"{safe_name}.mem"

        write_mem(mem_path, quantized, endianness)
        if write_numpy:
            np.save(target_dir / f"{safe_name}.npy", quantized)

        manifest["tensors"].append(
            tensor_manifest(
                name,
                array,
                quantized,
                saturated_count,
                mem_path,
                base_word_address,
            )
        )
        all_values.append(quantized.reshape(-1))
        base_word_address += int(quantized.size)

    if all_values:
        concatenated = np.concatenate(all_values)
        write_mem(target_dir / "all_weights.mem", concatenated, endianness)
        manifest["all_weights_mem_file"] = "all_weights.mem"
        if write_numpy:
            np.savez(
                target_dir / "weights_q8_8.npz",
                **{
                    sanitize_name(item["name"]): all_values[index].reshape(item["shape"])
                    for index, item in enumerate(manifest["tensors"])
                },
            )

    manifest["total_word_count"] = base_word_address
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest


def main() -> None:
    args = parse_args()
    inputs = args.inputs or default_inputs()
    if not inputs:
        raise SystemExit("No .npz or checkpoint inputs found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in inputs:
        manifest = export_input(path, args.output_dir, args.endianness, args.write_numpy)
        summaries.append(
            {
                "source": manifest["source"],
                "output": str(args.output_dir / sanitize_name(Path(manifest["source"]).stem)),
                "total_word_count": manifest["total_word_count"],
                "saturated_count": sum(
                    tensor["saturated_count"] for tensor in manifest["tensors"]
                ),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="ascii")
    for item in summaries:
        print(
            f"{item['source']} -> {item['output']} "
            f"words={item['total_word_count']} saturated={item['saturated_count']}"
        )


if __name__ == "__main__":
    main()

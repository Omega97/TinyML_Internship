#!/usr/bin/env python3
"""
Lab-ready script: Prepare a TinyML chess model for deployment on Arduino / XIAO ESP32-S3.

Pipeline:
  1. Load model (from checkpoint .pt or random for testing)
  2. Export / convert to TensorFlow Lite (.tflite), with optional quantization
  3. Generate Arduino-compatible C header containing the flatbuffer model data
     (ready to #include in a TFLite Micro sketch)

Usage examples:
  # Basic (float32 tflite + C header) from existing checkpoint
  python scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1

  # With dynamic-range quantization (recommended starting point for MCU)
  python scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1 --quantize dynamic

  # Full int8 (requires representative data for best accuracy)
  python scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1 --quantize int8

  # Just convert an already-exported .tflite into an Arduino header
  python scripts/prepare_for_arduino.py --from-tflite models/exported/tiny_policy_v0.1.tflite

Outputs are placed under models/exported/ and models/arduino/models/
"""

import argparse
import sys
from pathlib import Path

# Ensure we can import the project package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from src.tinymlinternship.config.settings import (
    CHECKPOINTS_DIR,
    EXPORTED_DIR,
    ARDUINO_MODELS_DIR,
)
from src.tinymlinternship.models.policy import TinyPolicy


def ensure_dirs():
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    ARDUINO_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str) -> torch.nn.Module:
    """Load TinyPolicy from checkpoint or create random one."""
    model = TinyPolicy(hidden_channels=16)
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}.pt"

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}. Using random weights.")
    model.eval()
    return model


def export_to_tflite(
    model: torch.nn.Module,
    model_name: str,
    quantize: str = "none",
) -> Path:
    """
    Export the model to .tflite with optional quantization.

    quantize options:
      - "none": float32 (default for debugging)
      - "dynamic": post-training dynamic range quantization (good size/speed win, easy)
      - "int8": full integer quantization (best for MCU; needs representative dataset)
    """
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    dummy_input = torch.randn(1, 12, 8, 8)

    # 1. TorchScript trace
    traced = torch.jit.trace(model, dummy_input)

    # 2. ONNX (stable intermediate)
    onnx_path = EXPORTED_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        traced,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"✅ Exported ONNX: {onnx_path}")

    # 3. ONNX → TensorFlow saved model (via onnx-tf)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = EXPORTED_DIR / f"{model_name}_saved_model"
    tf_rep.export_graph(str(saved_model_dir))

    # 4. Convert to TFLite + apply quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("🔧 Applying dynamic-range quantization")
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Representative dataset is critical for good int8 accuracy
        def representative_dataset():
            # TODO: replace with real board positions from your dataset for production
            for _ in range(50):
                # Random but valid-shaped input (in real use: fen_to_tensor results)
                data = np.random.randn(1, 12, 8, 8).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
        print("🔧 Applying full INT8 quantization (using random representative data)")

    tflite_model = converter.convert()

    tflite_path = EXPORTED_DIR / f"{model_name}.tflite"
    tflite_path.write_bytes(tflite_model)

    size_kb = tflite_path.stat().st_size / 1024
    print(f"🎉 TFLite model ready: {tflite_path} ({size_kb:.1f} KB)")

    return tflite_path


def tflite_to_c_header(
    tflite_path: Path,
    variable_name: str = "g_chess_model",
    header_path: Path | None = None,
) -> Path:
    """
    Convert a .tflite flatbuffer into a C header that can be included
    directly in an Arduino / TFLite Micro sketch.

    Produces:
        const unsigned char g_chess_model[] = { 0x1c, 0x00, ... };
        const unsigned int  g_chess_model_len = 12345;
    """
    if header_path is None:
        header_path = ARDUINO_MODELS_DIR / f"{tflite_path.stem}_model.h"

    data = tflite_path.read_bytes()
    model_len = len(data)

    # Build C array content (12 bytes per line is readable)
    lines = []
    for i in range(0, model_len, 12):
        chunk = data[i : i + 12]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_values}")

    array_content = ",\n".join(lines)

    header_content = f'''// Auto-generated by scripts/prepare_for_arduino.py
// Source: {tflite_path.name}
// DO NOT EDIT MANUALLY — re-run the script after retraining / requantizing.

#ifndef CHESS_MODEL_DATA_H_
#define CHESS_MODEL_DATA_H_

#include <cstdint>

const unsigned char {variable_name}[] = {{
{array_content}
}};

const unsigned int {variable_name}_len = {model_len};

#endif  // CHESS_MODEL_DATA_H_
'''

    header_path.write_text(header_content)
    print(f"✅ Arduino C header generated: {header_path}")
    print(f"   Variable: {variable_name}  |  Size: {model_len} bytes")

    return header_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a chess policy model for Arduino / XIAO deployment."
    )
    parser.add_argument(
        "--model-name",
        default="tiny_policy_v0.1",
        help="Name of the checkpoint (without .pt). Used to locate models/checkpoints/<name>.pt",
    )
    parser.add_argument(
        "--from-tflite",
        type=Path,
        default=None,
        help="Skip PyTorch export and use this existing .tflite file directly.",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "dynamic", "int8"],
        default="dynamic",
        help="Quantization mode. 'dynamic' is the best quick win for MCUs.",
    )
    parser.add_argument(
        "--variable-name",
        default="g_chess_model",
        help="Name of the C array variable in the generated header.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.from_tflite:
        tflite_path = args.from_tflite.resolve()
        if not tflite_path.exists():
            print(f"❌ TFLite file not found: {tflite_path}")
            sys.exit(1)
        print(f"Using existing TFLite: {tflite_path}")
    else:
        model = load_model(args.model_name)
        tflite_path = export_to_tflite(model, args.model_name, quantize=args.quantize)

    # Always produce (or update) the Arduino-ready header
    header_path = tflite_to_c_header(
        tflite_path,
        variable_name=args.variable_name,
    )

    print("\n" + "=" * 60)
    print("🚀 MODEL READY FOR ARDUINO / XIAO")
    print("=" * 60)
    print(f"TFLite file : {tflite_path}")
    print(f"C header    : {header_path}")
    print("\nIn your Arduino sketch, add:")
    print(f'  #include "{header_path.name}"')
    print(f"  // Then use {args.variable_name} and {args.variable_name}_len")
    print("\nNext steps:")
    print("  1. Copy the header into your Arduino project (or use full path)")
    print("  2. Add the TensorFlow Lite for Microcontrollers library")
    print("  3. Implement input tensor building (see featurizer.py for spec)")
    print("  4. Run inference and apply legal mask on the 4096 outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()

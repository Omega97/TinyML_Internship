"""
Export TinyPolicy to TensorFlow Lite / LiteRT format.

NOTE: For lab deployment on Arduino/XIAO, prefer the more complete script:
    python scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1 --quantize dynamic

This script also generates the C header ready to include in a TFLite Micro sketch.
"""
import onnx
from onnx_tf.backend import prepare
import torch
from src.tinymlinternship.config.settings import CHECKPOINTS_DIR
from src.tinymlinternship.models.policy import TinyPolicy
from src.tinymlinternship.config.settings import EXPORTED_DIR


def export_to_tflite(model_name: str = "tiny_policy_v0.1"):
    # 1. Load the PyTorch model
    model = TinyPolicy(hidden_channels=16)

    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}.pt"  # adjust if needed
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print("⚠️ Using random weights (no checkpoint found)")

    model.eval()

    # 2. Create dummy input (same shape used in training)
    dummy_input = torch.randn(1, 12, 8, 8)  # (batch, channels, height, width)

    # 3. Convert to TorchScript (traced)
    traced_model = torch.jit.trace(model, dummy_input)

    # 4. Export to ONNX (intermediate step - often more stable)
    onnx_path = EXPORTED_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        traced_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Exported to ONNX: {onnx_path}")

    # 5. Convert ONNX → TFLite (recommended way)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tflite_path = EXPORTED_DIR / f"{model_name}.tflite"
    tf_rep.export_graph(str(tflite_path))

    print(f"🎉 SUCCESS! Model exported to: {tflite_path}")
    print(f"   Size: {tflite_path.stat().st_size / 1024:.1f} KB")

    return tflite_path


if __name__ == "__main__":
    export_to_tflite()

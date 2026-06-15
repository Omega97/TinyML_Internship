# Export Pipeline (Bare Essentials)

This document describes the **minimal steps** to go from a model definition in code to something that can be loaded and run on the target hardware (e.g. Wio Terminal D51R or Seeed XIAO).

Full pipeline at [scripts/run_pipeline.py](scripts/run_pipeline.py).

#todo add the quantization/pruning steps.

---

## 1. Create the Base Model

Create the base model using the project's model code. Produce a PyTorch checkpoint (`.pt` file containing `state_dict` or a full model).

### Ultra-tiny value net (recommended for Wio Terminal D51R)
```bash
py -3.12 -c '
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import torch
from tinymlinternship.models.value import UltraTinyValueMLP

model = UltraTinyValueMLP().eval()
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "models/checkpoints/my_tiny_model.pt")
print("✅ Created models/checkpoints/my_tiny_model.pt")
'
```

### Policy net (larger, original TinyPolicy)
See the `create_and_save_random_policy()` function and `main()` entrypoint in [src/tinymlinternship/models/policy.py](src/tinymlinternship/models/policy.py).

**Output:**
- `models/checkpoints/my_tiny_model.pt` (or `tiny_value_wio.pt`, etc.)

The model is now in a standard PyTorch format and ready for export.

---

## 2. Export to a Portable Intermediate Format

Convert the PyTorch model to TorchScript (preferred for direct embedding) or ONNX.

### Explicit command (TorchScript)
From the checkpoint produced in step 1:

```bash
py -3.12 -c '
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import torch
from tinymlinternship.models.value import UltraTinyValueMLP   # use policy.TinyPolicy for the larger net

model = UltraTinyValueMLP().eval()
model.load_state_dict(torch.load("models/checkpoints/my_tiny_model.pt", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 768)   # 768 for value nets; use torch.randn(1, 12, 8, 8) for policy
traced = torch.jit.trace(model, dummy)

Path("models/exported").mkdir(parents=True, exist_ok=True)
traced.save("models/exported/my_tiny_model.ts.pt")
print("✅ Created models/exported/my_tiny_model.ts.pt")
'
```

The full export logic (including ONNX) is also implemented inside:

- [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py) (general case)
- [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) (Wio Terminal focused, also handles value nets)

**Output files:**
- `models/exported/my_tiny_model.ts.pt` (preferred for embedding)
- `models/exported/my_tiny_model.onnx` (+ `.onnx.data` if the model is large)

---

## 3. Prepare the Hardware Artifact (C Header / Blob)

Turn the exported binary into a C header that can be compiled into an Arduino sketch or stored on external flash.

Use the dedicated utility:

[scripts/bin_to_c_header.py](scripts/bin_to_c_header.py)

Example invocation (run from the project root):
```bash
py -3.12 scripts/bin_to_c_header.py \
    models/exported/my_tiny_model.ts.pt \
    --var-name g_chess_model \
    --out models/arduino/models/my_tiny_model.h
```

This produces `models/arduino/models/my_tiny_model.h` containing the model as a C byte array (see the script for the exact output format).

You can now:
- `#include "my_tiny_model.h"` directly in your `.ino` / `.cpp` files (the data goes into flash).
- Or extract just the raw binary (`my_tiny_model.ts.pt` or the `.bin` produced by some prepare scripts) and store it on the Wio Terminal's **4 MB external SPI flash** at runtime (better for larger models or when you want to update the model without re-flashing firmware).

**Scripts that wrap this step:**
- [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py)
- [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) (also produces a small custom `.bin` blob before the header)

---

## 4. Load the Model on Hardware

### In your Arduino sketch (Wio Terminal or XIAO)

See the sketch examples and TFLM integration patterns in [PRIVATE/load-model-howto.md](PRIVATE/load-model-howto.md) (section "Using the Model in an Arduino Sketch" and the Wio-specific notes).

The generated header (from Step 3) provides `g_xxx_model` and `g_xxx_model_len` that you pass to `tflite::GetModel()` or a custom loader.

### For the Wio Terminal D51R specifically (4 MB external flash)

Because internal flash is only 512 KB:
1. Keep the generated `.h` only for very small models.
2. For anything bigger than a few tens of KB, copy the **raw binary** (`.ts.pt` or the compact `.bin` from the prepare script) to the external QSPI/SPI flash.
3. At boot (or on demand), read the blob from external flash into a buffer and pass the pointer to `GetModel()` or your custom loader.

See `NOTES/hardware.md` (Wio Terminal section) for board-specific details.

### Minimal verification
- Flash a sketch that loads the model and runs inference on a known input (e.g. starting position).
- Compare the output (logits or value) against the same input run on the laptop using the original PyTorch model.
- Once parity is confirmed, you can add search, I/O, power measurement, etc.

---

## Summary of the Bare Pipeline

1. **Create base model** → `models/checkpoints/my_tiny_model.pt`  
   (run the one-liner in step 1 above using [src/tinymlinternship/models/value.py](src/tinymlinternship/models/value.py) or [src/tinymlinternship/models/policy.py](src/tinymlinternship/models/policy.py), or the prepare scripts)

2. **Export to TorchScript** → `models/exported/my_tiny_model.ts.pt`  
   (run the one-liner in step 2 above, or via [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py) / [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py))

3. **Generate hardware artifact** → `models/arduino/models/my_tiny_model.h`  
   (run the command in step 3 above using [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py))

4. **Load on device** → `#include` the header (or read the raw binary from external flash) and pass the bytes to your inference runtime / custom loader. (See [PRIVATE/load-model-howto.md](PRIVATE/load-model-howto.md))

---

## Key Scripts & Files

| Step              | Script / File                                                                                                                             | Output Location                |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Create model      | [src/tinymlinternship/models/policy.py](src/tinymlinternship/models/policy.py)<br>[src/tinymlinternship/models/value.py](src/tinymlinternship/models/value.py) | `models/checkpoints/`          |
| Export            | [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py)<br>[scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py)            | `models/exported/`             |
| Header generation | [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py)                                                                                  | `models/arduino/models/`       |
| Full Wio pipeline | [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py)                                                                                | All of the above + int8 `.bin` |
| Detailed how-to   | [PRIVATE/load-model-howto.md](PRIVATE/load-model-howto.md)                                                                                | —                              |

---

## Next Steps (when you are ready)

Once the bare flow works:
- Add training (`scripts/prepare_wio_tiny.py --train`)
- Add quantization (the prepare scripts have `--quantize` options)
- Add pruning
- Optimization
- Store models on the Wio's external 4 MB flash
- Precise power measurement
- Implement inference + search on the device
- Advanced deployment (Edge Impulse, SenseCraft, etc.)

This document will be expanded later to include those stages. For now it is intentionally minimal so you can get a model onto the hardware as quickly as possible in the lab.

---

#core 

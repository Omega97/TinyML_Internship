# Export Pipeline (Bare Essentials)

This document describes the **minimal steps** to go from a model definition in code to something that can be loaded and run on the target hardware (e.g. Wio Terminal D51R or Seeed XIAO).

#todo add the quantization/pruning steps.

---

## 1. Download the Base Model 

Download the base model using the project's model scripts. Produce a PyTorch checkpoint (`.pt` file containing `state_dict` or a full model).

- Policy models (original TinyPolicy): [models/policy.py](src/tinymlinternship/models/policy.py) — see `create_and_save_random_policy()` and the `main()` entrypoint.
- Ultra-tiny value nets (recommended for Wio Terminal D51R): [models/value.py](src/tinymlinternship/models/value.py) — `TinyValueMLP`, `UltraTinyValueMLP`, and the `create_tiny_value()` helper.

**Output:**
- `models/checkpoints/my_tiny_model.pt` (or `tiny_value_wio.pt`, etc.)

The model is now in a standard PyTorch format and ready for export.

---

## 2. Export to a Portable Intermediate Format

Convert the PyTorch model to ONNX or TorchScript. These formats are easier to consume in conversion tools or for direct embedding.

The export logic (TorchScript + ONNX) lives in the prepare scripts:

- [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py) (general case)
- [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) (Wio Terminal focused, also handles value nets)

**Output files:**
- `models/exported/my_tiny_model.ts.pt` (preferred for embedding)
- `models/exported/my_tiny_model.onnx` (+ `.onnx.data` if the model is large)

---

## 3. Prepare the Hardware Artifact (C Header / Blob)

Turn the exported binary into something that can be compiled into an Arduino sketch or stored on external flash.

Use the dedicated utility:

[scripts/bin_to_c_header.py](scripts/bin_to_c_header.py)

Example invocation (run from the project root):
```bash
py -3.12 scripts/bin_to_c_header.py \
    models/exported/my_tiny_model.ts.pt \
    --var-name g_chess_model \
    --out models/arduino/models/my_tiny_model.h
```

This produces a header in `models/arduino/models/` containing the model as a C byte array (see the script for the exact output format).

You can now:
- `#include "my_tiny_model.h"` directly in your `.ino` / `.cpp` files (the data goes into flash).
- Or extract just the raw binary (`my_tiny_model.ts.pt` or the `.bin` produced by some prepare scripts) and store it on the Wio Terminal's **4 MB external SPI flash** at runtime (better for larger models or when you want to update the model without reflashing firmware).

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

1. **Obtain base model** → `models/checkpoints/xxx.pt`  
   (via functions in [src/tinymlinternship/models/policy.py](src/tinymlinternship/models/policy.py) or [src/tinymlinternship/models/value.py](src/tinymlinternship/models/value.py), or the prepare scripts)

2. **Export** → `models/exported/xxx.ts.pt` (and/or `.onnx`)  
   (see export logic in [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py) and [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py))

3. **Generate hardware artifact** → `models/arduino/models/xxx_model.h` (or raw `.bin`)  
   (via [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py))

4. **Load on device** → `#include` the header (or read from external flash) and pass the bytes to your inference runtime / custom loader. (See [PRIVATE/load-model-howto.md](PRIVATE/load-model-howto.md))

---

## Key Scripts & Files

| Step              | Script / File                                                                 | Output Location                  |
|-------------------|-------------------------------------------------------------------------------|----------------------------------|
| Create model      | [src/tinymlinternship/models/policy.py](src/tinymlinternship/models/policy.py)<br>[src/tinymlinternship/models/value.py](src/tinymlinternship/models/value.py) | `models/checkpoints/`            |
| Export            | [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py)<br>[scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) | `models/exported/`               |
| Header generation | [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py)                      | `models/arduino/models/`         |
| Full Wio pipeline | [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py)                    | All of the above + int8 `.bin`   |
| Detailed how-to   | [PRIVATE/load-model-howto.md](PRIVATE/load-model-howto.md)                    | —                                |

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

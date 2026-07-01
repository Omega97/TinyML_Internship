# Pre-SARDINE archive

Code from the original TinyML export pipeline (768-feature value MLPs, policy nets, Wio int8 benchmarks) moved here so the active tree stays focused on [SARDINE](../../NOTES/SARDINE%20🐟.md).

## Contents

| Path | What |
|------|------|
| `Arduino/Wio_TinyValueTest/` | Wio Terminal value-net benchmark sketch |
| `src/tinymlinternship/` | Old `featurizer`, `value`, and `policy` modules |
| `scripts/` | Export, quantization, and Wio weight generation |
| `examples/` | Policy/value inference demos |
| `tests/test_policy_inference.py` | Policy inference tests |
| `models/` | Checkpoints, exports, and Arduino headers |
| `export_pipeline.md` | Old end-to-end export guide |

## Running legacy scripts

From the repo root, legacy Python scripts load both `src/` (config) and `legacy/pre-sardine/src/` (archived modules) via `bootstrap.py`:

```bash
py -3.12 legacy/pre-sardine/scripts/run_model.py --help
```

Arduino sketch: open `legacy/pre-sardine/Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino` in the IDE.
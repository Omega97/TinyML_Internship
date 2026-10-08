# World-Model Encoder (self-supervised L1) — Blueprint

> How to replace the standard L1 accumulator with a self-supervised "world
> model" latent, and how the **Train / Load** toggle exposes it in the MoE
> training UI. This is the implementable spec behind the UI "Encoder (L1)" group.

---

## 1. What the world model is

The standard NNUE uses a fixed feature extractor as L1:

$$a_{L1} = \text{CReLU}\big(W_{\text{acc}}\, x + b_{\text{acc}}\big),\qquad x \in \{0,1\}^{844}$$

The world model replaces only the **activation** of that same weight matrix with a
contrastively trained latent:

$$z = \text{SiLU}\big(W_{\text{WM}}\, x + b_{\text{WM}}\big),\qquad \hat z = \frac{z}{\lVert z\rVert_2}\ \text{(optional unit-hypersphere)}$$

Everything downstream is unchanged: the per-POV `z` is STM-ordered and
concatenated to `2W`, then `L2: 2W → H` (CReLU) and `head: H → 3` (softmax WDL).
Consequently `d_{L1} = W` (the accumulator width), and the state-dict keys stay
`l1.weight` / `l1.bias`, so **legacy checkpoints keep loading** with no migration.

### Toggle

- `encoder ∈ {standard, world_model}` (UI "Encoder (L1) → Type").
- `standard` = current behaviour, byte-for-byte.
- `world_model` = SiLU + (optional) unit-norm L1. It replaces L1 **globally** —
  the shared encoder, the router input, and every expert head all read `z`.
  Existing standard-L1 models are left exactly as they are.
- `world_model_source ∈ {train, load}` (UI "Encoder (L1) → Source").

| source | behaviour |
|---|---|
| `train` | contrastively train `W_WM` on mirror pairs (§3), then supervised-train `L2`+`head` with L1 frozen (§4); save the result as a world-model checkpoint (§5). |
| `load` | load an existing world-model checkpoint (`wm_checkpoint`), use it as-is. |

---

## 2. Positive pairs — board mirror symmetry

The dataset stores per-position FENs (no move/game structure), so the feasible
positive pair is the **board-mirror symmetry** already encoded by the dual-POV
encoder: for one position the white-POV and black-POV views are

$$(x,\ x^+) = (\text{white view},\ \text{black mirrored view})$$

Negatives are the other positions **in the same batch** (in-batch negatives).
No new data is required. (Consecutive-ply / transposition pairs are a future
option that needs a game-structured dataset and are out of scope here.)

Implementation: `encode_mirror_pair(base, batch, normalize)` in
`src/tinymlinternship/nnue/world_model.py` returns `(z_white, z_black)`.

---

## 3. Losses

- **InfoNCE** (`wm_loss = "infonce"`, symmetric SimCLR-style, temperature `wm_tau`):
  $$\mathcal L = \frac{1}{2}\Big[\text{CE}\big(\text{sim}(z,z^+)/\tau\big) + \text{CE}\big(\text{sim}(z^+,z)/\tau\big)\Big],\quad \text{sim}(u,v)=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}$$
- **VICReg** (`wm_loss = "vicreg"`, invariance `sim_coeff=25`, variance `var_coeff=25`,
  covariance `cov_coeff=1`, `gamma = wm_vicreg_gamma`):
  $$\mathcal L = \text{sim}\cdot \lVert z-z^+\rVert^2 + \text{var}\cdot\big[\mathcal L_{\text{var}}(z)+\mathcal L_{\text{var}}(z^+)\big] + \text{cov}\cdot\big[\mathcal L_{\text{cov}}(z)+\mathcal L_{\text{cov}}(z^+)\big]$$
  with $\mathcal L_{\text{var}} = \frac1{d}\sum_j \max(0,\gamma-\sqrt{\text{Var}(Z_{:,j})+\epsilon})$ and
  $\mathcal L_{\text{cov}} = \frac1{d}\sum_{i\ne j} C_{ij}^2$ on the batch covariance $C$.

Both live in `world_model.py` (`info_nce_loss`, `vicreg_loss`).

---

## 4. Collapse prevention & diagnostics

Collapse (constant or rank-deficient `z`) is countered by: InfoNCE's negative
repulsion (§3), VICReg's variance/covariance terms, and optional hypersphere
normalization (`wm_normalize`, on by default).

`collapse_diagnostics(z)` returns the singular-value spectrum and an
entropy-based **effective rank**; these are reported in the UI world-model stage
and written to the run summary. A healthy encoder has effective rank ≈ `W`; a
collapsed one ≈ 1.

---

## 5. Training / loading flow (implemented in `runner.py`)

1. Resolve the base architecture (`W = h`, `H`).
   - `world_model_source = load`: `load_dual_hidden_checkpoint(wm_checkpoint)` and
     **fail loudly if the checkpoint is not a world model**.
   - `world_model_source = train`: load a base checkpoint for L2/head init
     (`base_source = load`) or start fresh (`base_source = new`).
2. `train_world_model_encoder(...)`: train only `base.l1` with InfoNCE/VICReg on
   mirror pairs, set `base.encoder = "world_model"` + `base.normalize_l1`.
3. `_train_new_base(..., freeze_l1=True)`: supervised-train `L2`+`head` (L1 frozen).
4. Save the world-model base to `models/checkpoints/nnue/worldmodel_h{W}_H{H}/best.pt`
   (payload carries `encoder: "world_model"`, `normalize_l1`, `config.json`).
5. Downstream MoE proceeds unchanged: `DualHiddenMoE` / `SoftGatedMoE` clone the
   L2/head and share the world-model L1 activation (`l1_activation`).

---

## 6. Config schema (`TrainingConfig` fields)

| field | default | meaning |
|---|---|---|
| `encoder` | `standard` | `standard` \| `world_model` |
| `world_model_source` | `train` | `train` \| `load` |
| `wm_checkpoint` | `""` | path to a world-model `best.pt` |
| `wm_loss` | `infonce` | `infonce` \| `vicreg` |
| `wm_tau` | `0.1` | InfoNCE temperature |
| `wm_epochs` | `5` | contrastive epochs |
| `wm_lr` | `1e-3` | contrastive LR |
| `wm_normalize` | `True` | unit-hypersphere normalize `z` |
| `wm_vicreg_gamma` | `1.0` | VICReg variance target |

---

## 7. Artifacts

- World-model checkpoint: `models/checkpoints/nnue/worldmodel_h{W}_H{H}/best.pt`
  (+ `config.json`). Loadable by the UI and by `load_dual_hidden_checkpoint`.
- Run outputs (same as every MoE run): `RESULTS/moe/<run_name>/{summary.json,
  eval.json, labels*.npy, centroids.npy, diagnostics.json, moe.pt, ...}`.
  `summary.json` records the world-model config and collapse diagnostics.
- UI stage: `world_model` (per-epoch contrastive loss) + `base` (supervised head CE).

---

## 8. Evaluation (downstream)

The world-model base is scored exactly like the standard base: global test CE/MAE
(`eval.json`), and the same MoE battery (cluster → dispatcher → expert fine-tune →
`moe_vs_base_ce`). The comparison of interest is **world-model L1 vs standard L1**
at equal `W`/`H`, plus the collapse diagnostics (§4).

# Run `run_idrid_fidelity_all69.py` on Your GPU System — Runbook

**Purpose:** Reviewer 1, Comment 2 (manuscript `anujr_a-521`, second round) — re-run the exact
fidelity protocol of Algorithm 3 on **all 69 held-out IDRiD test images** (including the 18
misclassified ones) and report Deletion / Insertion AUC for **both** n = 69 (all images) and
n = 51 (correctly classified only), so the two yellow placeholders in the Subset-sensitivity
paragraph of the revised manuscript can be filled.

> ⚠️ **Use THIS copy of the script.** It contains 3 bug fixes that were found by executing the
> script end-to-end. Earlier copies crash. Verify the SHA-256 before running:
> ```
> c56d24ff10d7e63d73362e257e749d876104e9a93222f364e4e91b70854ca0a4  run_idrid_fidelity_all69.py
> ```
> Fixes in this version:
> 1. `load_model()` referenced an undefined `NUM_CLASSES` → `NameError` at startup (now uses `num_classes`).
> 2. Grad-CAM hooked `model.layers[-1].blocks[-1].norm1`, which does not exist in torchvision's
>    Swin (`AttributeError`); now hooks `model.norm` and reshapes the `(B,7,7,C)` token grid to
>    `(B,49,C)` so the 7×7 CAM is built correctly.
> 3. The Wilcoxon stage compared each method **against itself** instead of the Random baseline
>    (all-zero differences → guaranteed `ValueError` after the full run on any dataset); now
>    compares against `random_deletion_auc` / `random_insertion_auc`, with a guard that returns
>    `NaN` instead of crashing on degenerate samples.

---

## 1. Requirements

- Python ≥ 3.10, CUDA GPU (any ≥ 6 GB VRAM; tested path also runs CPU-only, just slower)
- Packages:

```bash
pip install torch torchvision lime scikit-image scipy numpy pillow
```

GPU check (optional):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If CUDA is unavailable the script automatically falls back to CPU
(~40–50 min on an RTX 4070 Laptop GPU vs. ~4–6 h on a typical CPU).

---

## 2. Input 1 — the 69-image IDRiD test split

Create a folder with one subfolder per class (exact class-name spelling matters):

```
idrid_test/
├── No_DR/            20 images   (*.jpg / *.png / *.jpeg)
├── Mild/              3 images
├── Moderate/         24 images
├── Severe/           13 images
└── Proliferate_DR/    9 images
```

- These are the **69 held-out test images from your local 455-image IDRiD working copy**
  (the same split that produced Table 3 / the confusion matrix). Copy them from your
  dataset; do not resize — the script normalizes to 224×224 internally.
- The script prints `[INFO] found N test images (expected 69: 20 No_DR, 3 Mild, 24 Moderate,
  13 Severe, 9 Prolif)` — the count is informational; per-class assignment comes from the
  folder names.

---

## 3. Input 2 — the fine-tuned checkpoint

Place your IDRiD-fine-tuned Swin-T weights anywhere, e.g. `models/swin_t_idrid_finetuned.pth`.
The loader accepts either:

- a **plain `state_dict`** (`torch.save(model.state_dict(), path)`), or
- a dict with a `'state_dict'` key.

The architecture it must match is torchvision `swin_t` with the head replaced by
`nn.Linear(768, 5)`.

If you are unsure what your checkpoint contains, inspect it first:

```python
import torch
sd = torch.load('models/swin_t_idrid_finetuned.pth', map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
ks = list(sd.keys())
print(len(ks)); print(ks[:5]); print(ks[-5:])
```

- Keys ending in `head.weight` with shape `[5, 768]` → good to go.
- Keys with a redundant prefix (`module.`, `model.`) → strip it:

```python
sd = {k.removeprefix('module.').removeprefix('model.'): v for k, v in sd.items()}
torch.save(sd, 'models/swin_t_clean.pth')
```

- If your checkpoint was produced with **timm** or a different Swin implementation, keys will
  not match torchvision's; easiest fix is to rebuild the head on torchvision swin_t and load
  the matching backbone weights, or re-export the state dict from your training code.

---

## 4. Run

```bash
python run_idrid_fidelity_all69.py \
    --images idrid_test \
    --checkpoint models/swin_t_idrid_finetuned.pth \
    --out idrid69_results
```

Arguments: `--images` (class-folder root), `--checkpoint` (path above), `--out` (results dir;
created if missing).

Protocol is fixed inside the script and replicates the repo's
`run_fidelity_full_3methods.py` / Algorithm 3: SLIC `n_segments=100, compactness=10, sigma=1,
start_label=1`; LIME `num_samples=1000, hide_color=0, top_labels=1, random_state=42`;
Deletion/Insertion 20 steps with constant ImageNet-mean baseline (123.675, 116.28, 103.53);
Random baseline = 5 shuffles. Methods: Standard LIME, Grad-CAM, Guided-LIME (α = 1.0), Random.

You can watch progress in `idrid69_results/progress.json` (written after every image) or the
console (`[10/69] elapsed …`).

Determinism: seed 42 everywhere. GPU reduction order may shift AUCs in the 3rd–4th decimal
place relative to a CPU run — irrelevant for reporting to 4 decimals.

---

## 5. What you get

- `idrid69_results/idrid_fidelity_n69_and_n51.json` — per-image records (predicted class,
  correct flag, all six AUCs + random), aggregate means ± std for both subsets, Wilcoxon tests.
- `idrid69_results/progress.json` — running per-image records.
- A console block:

```
READY-TO-PASTE VALUES FOR THE MANUSCRIPT PLACEHOLDERS
-- n = 69 (all test images) --          Deletion X.XXXX ± X.XXX   Insertion Y.YYYY ± Y.YYY
-- n = 51 (correctly classified only) --  …
-- Wilcoxon --  …
```

Sanity checks before trusting the numbers:
1. `n_correct_only` in the JSON should be ≈ 51 (69 − 18 misclassified, matching your confusion matrix).
2. The n = 51 row for Standard LIME should land near the Table 9 values (≈ 0.3412 / 0.6506);
   small drift from re-measuring is fine, a large gap means the checkpoint or split differs.
3. Random Baseline should sit around 0.5 on both metrics — far worse than all real methods.

---

## 6. Filling the manuscript placeholders

Replace the two yellow placeholders in the Subset-sensitivity paragraph with:

> "The aggregate Deletion/Insertion AUC under the n = 69 protocol are [Guided-LIME:
> `<del>` / `<ins>`; Standard LIME: `<del>` / `<ins>`], and the resulting method ranking is
> `[ranking]`."

and report the n = 51 comparison immediately after, noting whether the ranking and the
Guided-vs-LIME relationship are preserved when the 18 misclassified images are included.
State that the Wilcoxon signed-rank tests vs. the random baseline remain significant
(paste the p-values from the console block). Then delete the legend box at the top of the
manuscript.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: missing class folder: …` | One of the 5 class subfolders is missing or misspelled (exact names above). |
| `load_state_dict … size mismatch` / `Missing key(s)` | Checkpoint architecture ≠ torchvision swin_t with a 5-class head. Inspect keys (Section 3) and re-export. |
| `CUDA out of memory` | The script batches at 128; if VRAM < 6 GB, lower `batch_size=128` in `batched_predict()` and `generate_lime()` to 64. |
| Wilcoxon `NaN` p-values | Degenerate subset (e.g., a method exactly ties the random baseline on every image). With the real 69-image data this should not occur. |
| `np.trapz`/`np.trapezoid` warning | None expected; the script auto-selects the right numpy call. |
| Slow LIME progress bar | Normal — 1000 perturbation samples per image dominate the runtime. |

---

## 8. Optional extensions

- **RISE / SHAP / Integrated Gradients / Attention Rollout:** implement each with the same
  per-image interface (return a superpixel→score dict), register it in `METHOD_REGISTRY`, and
  add the AUC computation in `evaluate_image()` to reproduce the full 7-method comparison on
  n = 69 (useful if reviewers ask in a later round).
- **Official 103-image test set:** if you decide to report the official IDRiD test split
  instead of the local 69-image one, populate the class folders with all 103 images
  (34/5/32/19/13) and run identically — no code change needed; relabel the n in the text.

Dataset credit: IDRiD is distributed under CC-BY-4.0 (Porwal et al.); cite it as in the manuscript.


# Guided-LIME: A Hybrid Explainable AI Framework for Causally Faithful Medical Image Interpretation

This repository contains the experimentation code, figures, and result files for the paper **"Guided-LIME: A Hybrid Explainable AI Framework for Causally Faithful Medical Image Interpretation"** (submitted to *Journal of Computer Science and Technology*).

Guided-LIME is a hybrid post-hoc explainer that multiplicatively fuses the global spatial prior of Grad-CAM with the local superpixel precision of LIME:

```
S(s_i) = L(s_i) * (1 + G(s_i))
```

The framework is evaluated on the APTOS 2019 Blindness Detection dataset for Diabetic Retinopathy (DR) severity grading using a Swin Transformer backbone.

## Headline Results (full N=550 test set)

| Method | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---|
| Random Baseline | 0.5533 ± 0.178 | 0.5037 ± 0.177 |
| Grad-CAM | 0.4832 ± 0.200 | 0.5979 ± 0.178 |
| Standard LIME | 0.3412 ± 0.172 | 0.6506 ± 0.201 |
| **Guided-LIME (Ours)** | **0.3329 ± 0.169** | **0.6517 ± 0.198** |

**Wilcoxon signed-rank tests (N=550):**
- Guided-LIME vs Random (Deletion): p = 5.3 × 10⁻⁹²
- Guided-LIME vs Standard LIME (Deletion): p = 5.6 × 10⁻²⁵
- Guided-LIME vs Grad-CAM (Deletion): p = 1.4 × 10⁻⁷³
- Guided-LIME vs Grad-CAM (Insertion): p = 1.4 × 10⁻²⁵

Test Quadratic Weighted Kappa (QWK) on APTOS 2019: **0.8588**

## Repository Structure

```
Guided-Lime/
├── data/
│   ├── dataset_config.json          # Dataset + training + LIME config
│   └── test_split.json              # Deterministic 70/15/15 stratified split (550 test images)
│
├── src/
│   ├── run_fidelity_full_3methods.py # MAIN: full N=550 evaluation of Grad-CAM, Standard LIME, Guided-LIME, Random
│   ├── run_fidelity_full.py          # N=550 evaluation of Standard LIME + Random
│   ├── stability_analysis_full.py    # N=550 explanation-stability under noise + rotation perturbations
│   ├── gradcam_utils.py              # SwinGradCAM: Grad-CAM for Swin-T via hooks on model.layers[-1]
│   ├── hybrid_explainer.py           # Guided-LIME score computation: S = L * (1 + G)
│   ├── gradcam_fidelity.py           # Standalone Grad-CAM deletion/insertion script (pilot)
│   ├── hybrid_fidelity.py            # Standalone Guided-LIME deletion/insertion script (pilot)
│   ├── xai_utils.py                  # Shared XAI utilities (segmentation, masking, predict wrappers)
│   ├── run_clinical_analysis.py      # Jaccard clinical-alignment computation
│   ├── qualitative_viz.py            # Per-class and atlas qualitative visualisation
│   ├── run_hybrid_fidelity.py        # Wrapper for the hybrid 3-method evaluation
│   ├── 06_Statistical_Validation.py  # Wilcoxon + bootstrap CI computation
│   └── 05_Fidelity_Metrics.ipynb     # Jupyter notebook: high-level metric exploration
│
├── figures/
│   ├── architecture_diagram.jpeg     # End-to-end system architecture (6 modules)
│   ├── confusion_matrix.png          # 5-class confusion matrix (QWK = 0.8588)
│   ├── training_history.png          # 30-epoch training/validation loss and accuracy
│   ├── qualitative_atlas_full.png    # Per-class qualitative comparison of Grad-CAM vs Guided-LIME
│   ├── qualitative_*.png             # Per-class qualitative detail images
│   ├── fidelity_comparison_bars.png  # Bar chart: Deletion & Insertion AUC for all 4 methods
│   ├── gradcam_comparison_bars.png    # Standalone Grad-CAM comparison
│   ├── hybrid_comparison_bars.png    # Standalone hybrid comparison
│   ├── bootstrap_confidence_intervals.png  # 95% bootstrap CIs for LIME-vs-Random
│   ├── preprocessing_effect.png      # Pre-processing pipeline before/after
│   ├── lime_explanation_final.png    # LIME failure-mode case study
│   └── fidelity_curves_*.png         # Per-image Deletion + Insertion curves (sample)
│
└── results/
    ├── fidelity_full_results.json         # FULL N=550 results: LIME, Grad-CAM, Guided-LIME, Random
    ├── stability_full_results.json        # FULL N=550 stability under noise + rotation
    ├── gradcam_fidelity_results.json      # Pilot Grad-CAM evaluation
    ├── hybrid_fidelity_results.json       # Pilot Guided-LIME evaluation
    ├── hybrid_comparison_results.json     # Pilot hybrid comparison summary
    ├── fidelity_metrics_results.json      # Original (small-N) LIME-vs-Random pilot
    ├── clinical_alignment_results.json    # Jaccard IoU per DR class
    ├── stability_results.json             # Original (small-N) stability pilot
    ├── stability_detailed_results.csv     # Per-image stability table
    ├── statistical_validation_results.json # Wilcoxon + bootstrap CIs
    └── statistical_validation_report.txt   # Human-readable validation report
```

## Hardware & Software Requirements

The full N=550 evaluation was run on:
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM
- **CUDA:** 12.4
- **PyTorch:** 2.6.0+cu124
- **Python:** 3.13.3
- **LIME samples per image:** 1,000
- **Random baseline shuffles per image:** 5
- **Grad-CAM:** 1 forward + 1 backward pass via `SwinGradCAM`
- **Total wall time:** ~90 minutes for fidelity, ~90 minutes for stability

Total per-image explanation time is approximately 9–13 s on the above hardware.

## Quick Start: Reproducing the Headline Results

1. Place the APTOS 2019 dataset under `DR_XAI_Project/data/colored_images/{No_DR,Mild,Moderate,Severe,Proliferate_DR}/`.
2. Place the trained Swin-T checkpoint at `DR_XAI_Project/models/swin_transformer_final_generalized.pth`.
3. From the `DR_XAI_Project/` directory:

```bash
# Full N=550 fidelity (computes Grad-CAM, Standard LIME, Guided-LIME, Random)
python run_fidelity_full_3methods.py

# Full N=550 stability
python stability_analysis_full.py
```

Both scripts save per-image progress incrementally to JSON, so a partial run can be resumed safely.

## Statistical Methodology

- **Fidelity:** Deletion AUC and Insertion AUC with constant-gray baseline (ImageNet mean BGR).
- **Significance:** Two-sided Wilcoxon signed-rank test on paired per-image AUC values (N=550 pairs).
- **Confidence intervals:** 2,000-iteration bootstrap on the per-image mean.
- **Effect size:** $r = |Z| / \sqrt{N}$.

## License

MIT License. The trained model weights and the APTOS 2019 dataset are not included in this repository; please obtain them from their original sources.

## Citation

```bibtex
@article{acharya2026guidedlime,
  title   = {Guided-LIME: A Hybrid Explainable AI Framework for Causally
             Faithful Medical Image Interpretation},
  author  = {Acharya, Pratham and Hasan, Raza},
  journal = {Journal of Computer Science and Technology},
  year    = {2026},
  note    = {Submitted}
}
```

PROVENANCE & CAVEATS - IDRiD Real Run (run_idrid_fidelity_all69.py)
=====================================================================

WHAT WAS RUN
- download/run_idrid_fidelity_all69.py, UNMODIFIED protocol (SLIC 100/10/1,
  LIME K=1000 seed 42, hide_color=0, top_labels=1, 20-step curves, ImageNet-
  mean constant baseline, 5-shuffle random baseline), executed end-to-end
  in this sandbox (CPU, 2 cores, 227 min).

DATA (REAL)
- Official IDRiD "B. Disease Grading" test set: ALL 103 images
  (No_DR 34, Mild 5, Moderate 32, Severe 19, Proliferate_DR 13),
  organized from the official ground-truth CSV.
- Source: public HuggingFace mirror MahsaTorki/IDRiD_Dataset
  (dataset distributed CC-BY-4.0).

MODEL (PROXY - IMPORTANT)
- The paper's IDRiD-fine-tuned Swin-T checkpoint is a private author
  artifact and is NOT available here. A PROXY checkpoint was trained
  in-sandbox: torchvision swin_t, ImageNet init, 5 epochs on the official
  413-image train split, same 224x224 transform as the fidelity script.
- Proxy test performance: acc 52.4% (54/103 correct), macro F1 0.392.
- Therefore the AUC VALUES BELOW ARE NOT MANUSCRIPT VALUES. They are a
  faithful dress rehearsal of the protocol on real images + a real
  IDRiD-trained Swin-T, demonstrating exactly what the author's run
  will output (on their own 69-image split and their own weights).

HEADLINE RESULTS (proxy model)
- n = 103 (all images):   Deletion / Insertion AUC
  Standard LIME     0.2977 / 0.6044
  Grad-CAM          0.3388 / 0.6122
  Guided-LIME       0.2951 / 0.6076
  Random Baseline   0.5109 / 0.4865
- n = 54 (correct only):
  Standard LIME     0.2935 / 0.6583
  Grad-CAM          0.3821 / 0.6491
  Guided-LIME       0.2912 / 0.6606   <- best on BOTH metrics in this subset
  Random Baseline   0.5522 / 0.5277
- Wilcoxon (n=103): all methods vs Random p <= 2.5e-09 (both metrics);
  Guided vs Standard LIME deletion p = 0.024.
- Wilcoxon (n=54): all methods vs Random p <= 9e-07; Guided vs LIME
  deletion p = 0.231 (ns at this subset size).

FILES
- IDRiD_Real_Run_Results.json     full per-image records + summaries + tests
- IDRiD_Real_Run_Console_Log.txt  exact console output incl. READY-TO-PASTE block

AUTHOR NEXT STEP
- Run the same (already bug-fixed) script on YOUR machine with YOUR
  fine-tuned checkpoint and YOUR 69-image split (~40 min on RTX 4070),
  then paste the printed values into the two yellow manuscript
  placeholders. The sensitivity-analysis paragraph should then also state
  model identity, split definition, and the n=69 provenance.

"""
hybrid_fidelity.py
================================
Tier 3 (Optional but Strong): Hybrid LIME + Grad-CAM Explanation

Combines LIME importance scores and Grad-CAM per-superpixel mean activation
into a hybrid ranking:

    hybrid_score(s) = alpha * LIME_score(s) + (1 - alpha) * GradCAM_score(s)

Runs Deletion & Insertion tests for:
  - LIME-only ranking
  - Grad-CAM-only ranking
  - Hybrid ranking (alpha=0.5)

Then produces a 3-way comparison bar chart.

Writes: hybrid_fidelity_results.json
        hybrid_comparison_bars.png
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from skimage.segmentation import slic
from skimage.filters import gaussian as skimage_gaussian
from lime.lime_image import LimeImageExplainer
import warnings
warnings.filterwarnings('ignore')

from gradcam_utils import SwinGradCAM, get_transform, load_image

# ── CONFIG ───────────────────────────────────────────────────────────────────
SEED           = 42
ALPHA          = 0.5      # Weight for LIME vs Grad-CAM (0=GradCAM only, 1=LIME only)
NUM_STEPS      = 20
NUM_RAND_RUNS  = 3
NUM_SEGMENTS   = 100
LIME_SAMPLES   = 2000
MODEL_PATH     = 'models/swin_transformer_final_generalized.pth'
CONFIG_FILE    = 'dataset_config.json'
LIME_RESULTS   = 'fidelity_metrics_results.json'
OUTPUT_JSON    = 'hybrid_fidelity_results.json'
OUTPUT_FIG     = 'hybrid_comparison_bars.png'
DATA_DIR       = 'data/colored_images'
IMAGES_PER_CLASS = 2

FOLDER_TO_CLASS = {
    'No_DR':         'No_DR',
    'Mild':          'Mild',
    'Moderate':      'Moderate',
    'Severe':        'Severe',
    'Proliferate_DR':'Proliferative_DR',
}

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── DEVICE & MODEL ───────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

with open(CONFIG_FILE, 'r') as f:
    cfg = json.load(f)
NUM_CLASSES   = cfg['num_classes']
CLASS_MAPPING = cfg['class_to_idx']
idx_to_class  = {v: k for k, v in CLASS_MAPPING.items()}
class_names   = [idx_to_class[i] for i in range(NUM_CLASSES)]

model_nn = models.swin_t(weights=None)
model_nn.head = nn.Linear(model_nn.head.in_features, NUM_CLASSES)
model_nn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_nn.eval().to(device)

gcam = SwinGradCAM(model_nn, device)
tf   = get_transform()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def predict_fn(images):
    batch = torch.stack([tf(Image.fromarray(img.astype('uint8'))) for img in images]).to(device)
    with torch.no_grad():
        return torch.softmax(model_nn(batch), dim=1).cpu().numpy()


def get_lime_scores(image_pil, class_idx):
    img_np    = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=SEED)
    def seg_fn(img):
        return slic(img, n_segments=NUM_SEGMENTS, compactness=10, sigma=1, start_label=1)
    expl = explainer.explain_instance(
        img_np, predict_fn, top_labels=1,
        hide_color=0, num_samples=LIME_SAMPLES,
        batch_size=32, segmentation_fn=seg_fn
    )
    segments = expl.segments
    scores   = dict(expl.local_exp[class_idx])
    return segments, scores


def get_gradcam_scores(image_pil, segments, class_idx):
    img_np  = np.array(image_pil.resize((224, 224)))
    pil_224 = Image.fromarray(img_np)
    tensor  = tf(pil_224).unsqueeze(0)
    cam, _, _ = gcam.generate(tensor, class_idx=class_idx)
    scores = {int(seg_id): float(cam[segments == seg_id].mean())
              for seg_id in np.unique(segments)}
    return scores


def normalize_scores(scores_dict):
    """Min-max normalize scores dict to [0, 1]."""
    vals = np.array(list(scores_dict.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in scores_dict}
    return {k: float((v - mn) / (mx - mn)) for k, v in scores_dict.items()}


def run_deletion_insertion(image_pil, segments, importance_scores, class_idx):
    img_np  = np.array(image_pil.resize((224, 224))).astype('float32')
    blurred = (skimage_gaussian(img_np / 255.0, sigma=10,
                                channel_axis=-1) * 255.0).astype('float32')
    ordered   = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    seg_ids   = [s[0] for s in ordered]
    total     = len(seg_ids)
    per_step  = max(1, total // NUM_STEPS)

    def get_conf(modified):
        pil = Image.fromarray(modified.astype('uint8'))
        t = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.softmax(model_nn(t), dim=1)[0, class_idx].item()

    # Deletion
    mod_del = img_np.copy()
    del_scores = [get_conf(mod_del)]
    for step in range(NUM_STEPS):
        s, e = step * per_step, min((step + 1) * per_step, total)
        for idx in range(s, e):
            mod_del[segments == seg_ids[idx]] = blurred[segments == seg_ids[idx]]
        del_scores.append(get_conf(mod_del))
        if e >= total: break
    del_auc = np.trapezoid(del_scores, np.linspace(0, 1, len(del_scores))) if hasattr(np, 'trapezoid') else np.trapz(del_scores, np.linspace(0, 1, len(del_scores)))

    # Insertion
    mod_ins = blurred.copy()
    ins_scores = [get_conf(mod_ins)]
    for step in range(NUM_STEPS):
        s, e = step * per_step, min((step + 1) * per_step, total)
        for idx in range(s, e):
            mod_ins[segments == seg_ids[idx]] = img_np[segments == seg_ids[idx]]
        ins_scores.append(get_conf(mod_ins))
        if e >= total: break
    ins_auc = np.trapezoid(ins_scores, np.linspace(0, 1, len(ins_scores))) if hasattr(np, 'trapezoid') else np.trapz(ins_scores, np.linspace(0, 1, len(ins_scores)))

    return del_auc, ins_auc


# ── TEST IMAGES ───────────────────────────────────────────────────────────────

# Prefer reusing LIME-evaluated images
lime_data    = json.load(open(LIME_RESULTS))
lime_indiv   = lime_data.get('individual_results', [])
test_images  = [r['image_path'] for r in lime_indiv if os.path.exists(r['image_path'])][:6]

if len(test_images) < 3:
    rng2 = np.random.RandomState(SEED)
    for folder in FOLDER_TO_CLASS:
        cls_dir = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(cls_dir): continue
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        sel   = rng2.choice(files, min(IMAGES_PER_CLASS, len(files)), replace=False)
        test_images += [os.path.join(cls_dir, f) for f in sel]

print(f"Running hybrid evaluation on {len(test_images)} images…\n")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

all_results = []

for img_path in test_images:
    name = os.path.basename(img_path)
    print(f"  {name}")
    try:
        pil, tensor = load_image(img_path)
        tensor      = tensor.to(device)
        with torch.no_grad():
            probs     = torch.softmax(model_nn(tensor), dim=1)
            class_idx = int(probs.argmax(dim=1).item())
        print(f"    Class: {class_names[class_idx]}")

        # LIME scores
        print("    LIME…", end='', flush=True)
        segments, lime_sc = get_lime_scores(pil, class_idx)
        lime_norm = normalize_scores(lime_sc)
        del_lime, ins_lime = run_deletion_insertion(pil, segments, lime_norm, class_idx)
        print(f" Del={del_lime:.4f} Ins={ins_lime:.4f}")

        # Grad-CAM scores
        print("    Grad-CAM…", end='', flush=True)
        gcam_sc   = get_gradcam_scores(pil, segments, class_idx)
        gcam_norm = normalize_scores(gcam_sc)
        del_gcam, ins_gcam = run_deletion_insertion(pil, segments, gcam_norm, class_idx)
        print(f" Del={del_gcam:.4f} Ins={ins_gcam:.4f}")

        # Hybrid scores
        print("    Hybrid…", end='', flush=True)
        all_keys    = set(lime_norm.keys()) | set(gcam_norm.keys())
        hybrid_sc   = {k: ALPHA * lime_norm.get(k, 0.0) + (1 - ALPHA) * gcam_norm.get(k, 0.0)
                       for k in all_keys}
        del_hyb, ins_hyb = run_deletion_insertion(pil, segments, hybrid_sc, class_idx)
        print(f" Del={del_hyb:.4f} Ins={ins_hyb:.4f}")

        all_results.append({
            'image': name, 'class': class_names[class_idx],
            'lime_del': del_lime, 'lime_ins': ins_lime,
            'gcam_del': del_gcam, 'gcam_ins': ins_gcam,
            'hyb_del':  del_hyb,  'hyb_ins':  ins_hyb,
        })
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("HYBRID FIDELITY COMPARISON SUMMARY")
print("=" * 65)
header = f"{'Image':<22} {'Class':<20} {'LIME D':>7} {'GC D':>7} {'Hyb D':>7} {'LIME I':>7} {'GC I':>7} {'Hyb I':>7}"
print(header)
print("-" * 80)
for r in all_results:
    print(f"{r['image']:<22} {r['class']:<20} "
          f"{r['lime_del']:7.4f} {r['gcam_del']:7.4f} {r['hyb_del']:7.4f} "
          f"{r['lime_ins']:7.4f} {r['gcam_ins']:7.4f} {r['hyb_ins']:7.4f}")

avgs = {k: np.mean([r[k] for r in all_results]) for k in
        ['lime_del','gcam_del','hyb_del','lime_ins','gcam_ins','hyb_ins']}
print("-" * 80)
print(f"{'AVERAGE':<22} {'':<20} "
      f"{avgs['lime_del']:7.4f} {avgs['gcam_del']:7.4f} {avgs['hyb_del']:7.4f} "
      f"{avgs['lime_ins']:7.4f} {avgs['gcam_ins']:7.4f} {avgs['hyb_ins']:7.4f}")

# ── BAR CHART ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('3-Way Fidelity Comparison: LIME vs Grad-CAM vs Hybrid', fontsize=13, fontweight='bold')

methods = ['LIME', 'Grad-CAM', f'Hybrid (α={ALPHA})']
colors  = ['steelblue', 'darkorange', 'mediumseagreen']

for ax, keys, title, note in [
    (axes[0], ['lime_del','gcam_del','hyb_del'], 'Deletion AUC', '(Lower = Better)'),
    (axes[1], ['lime_ins','gcam_ins','hyb_ins'], 'Insertion AUC','(Higher = Better)'),
]:
    vals = [avgs[k] for k in keys]
    bars = ax.bar(methods, vals, color=colors, alpha=0.85)
    ax.set_title(f'{title}\n{note}', fontsize=12)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}',
                ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved: {OUTPUT_FIG}")

with open(OUTPUT_JSON, 'w') as f:
    json.dump({'summary': avgs, 'per_image': all_results, 'alpha': ALPHA}, f, indent=2)
print(f"✓ Results saved: {OUTPUT_JSON}")

gcam.remove_hooks()

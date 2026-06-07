"""
gradcam_fidelity.py
================================
Tier 2A: Grad-CAM Fidelity Comparison

Runs Deletion & Insertion tests using Grad-CAM superpixel ranking
and compares AUCs side-by-side against LIME.

Reads:  fidelity_metrics_results.json  (LIME results)
Writes: gradcam_fidelity_results.json
        gradcam_comparison_bars.png
        gradcam_comparison_table.txt
"""

import os, json, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from skimage.filters import gaussian as skimage_gaussian
import warnings
warnings.filterwarnings('ignore')

# ── Local imports ────────────────────────────────────────────────────────────
from gradcam_utils import SwinGradCAM, get_transform, load_image

# ── CONFIG ───────────────────────────────────────────────────────────────────
SEED           = 42
NUM_STEPS      = 20
NUM_RAND_RUNS  = 5
NUM_SEGMENTS   = 100
MODEL_PATH     = 'models/swin_transformer_final_generalized.pth'
CONFIG_FILE    = 'dataset_config.json'
LIME_RESULTS   = 'fidelity_metrics_results.json'
OUTPUT_JSON    = 'gradcam_fidelity_results.json'
OUTPUT_FIG     = 'gradcam_comparison_bars.png'
OUTPUT_TABLE   = 'gradcam_comparison_table.txt'
DATA_DIR       = 'data/colored_images'
IMAGES_PER_CLASS = 3   # Use same images as LIME where possible; otherwise 3/class

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
    config = json.load(f)
NUM_CLASSES   = config['num_classes']
CLASS_MAPPING = config['class_to_idx']
idx_to_class  = {v: k for k, v in CLASS_MAPPING.items()}
class_names   = [idx_to_class[i] for i in range(NUM_CLASSES)]

model = models.swin_t(weights=None)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)
print(f"✓ Model loaded")

gcam = SwinGradCAM(model, device)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_gradcam_importance(image_pil, class_idx):
    """
    Generate Grad-CAM heatmap and compute per-superpixel mean importance.
    Returns segments, importance_scores dict, cam_224 array.
    """
    img_np  = np.array(image_pil.resize((224, 224)))
    pil_224 = Image.fromarray(img_np)
    tensor  = get_transform()(pil_224).unsqueeze(0)

    cam_224, pred_class, _ = gcam.generate(tensor, class_idx=class_idx)

    # Superpixel segmentation (same as LIME for fair comparison)
    segments = slic(img_np, n_segments=NUM_SEGMENTS, compactness=10,
                    sigma=1, start_label=1)

    # Mean Grad-CAM value per segment
    importance_scores = {}
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        importance_scores[seg_id] = float(cam_224[mask].mean())

    return segments, importance_scores, cam_224


def run_deletion(image_pil, segments, importance_scores, class_idx, blur_sigma=10):
    img_np = np.array(image_pil.resize((224, 224))).astype('float32')
    blurred = (skimage_gaussian(img_np / 255.0, sigma=blur_sigma,
                                channel_axis=-1) * 255.0).astype('float32')

    ordered = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    seg_ids = [s[0] for s in ordered]
    total   = len(seg_ids)
    per_step = max(1, total // NUM_STEPS)

    tf     = get_transform()
    scores = []
    modified = img_np.copy()

    def conf():
        pil = Image.fromarray(modified.astype('uint8'))
        t = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(t), dim=1)
        return p[0, class_idx].item()

    scores.append(conf())
    for step in range(NUM_STEPS):
        start = step * per_step
        end   = min(start + per_step, total)
        for idx in range(start, end):
            mask = segments == seg_ids[idx]
            modified[mask] = blurred[mask]
        scores.append(conf())
        if end >= total:
            break

    x   = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return scores, auc


def run_insertion(image_pil, segments, importance_scores, class_idx, blur_sigma=10):
    img_np = np.array(image_pil.resize((224, 224))).astype('float32')
    blurred = (skimage_gaussian(img_np / 255.0, sigma=blur_sigma,
                                channel_axis=-1) * 255.0).astype('float32')

    ordered = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    seg_ids = [s[0] for s in ordered]
    total   = len(seg_ids)
    per_step = max(1, total // NUM_STEPS)

    tf       = get_transform()
    scores   = []
    modified = blurred.copy()

    def conf():
        pil = Image.fromarray(modified.astype('uint8'))
        t = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(t), dim=1)
        return p[0, class_idx].item()

    scores.append(conf())
    for step in range(NUM_STEPS):
        start = step * per_step
        end   = min(start + per_step, total)
        for idx in range(start, end):
            mask = segments == seg_ids[idx]
            modified[mask] = img_np[mask]
        scores.append(conf())
        if end >= total:
            break

    x   = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return scores, auc


# ── COLLECT TEST IMAGES ───────────────────────────────────────────────────────

# Try to reuse images from LIME results
lime_data     = json.load(open(LIME_RESULTS))
lime_results  = lime_data.get('individual_results', [])
test_images   = [r['image_path'] for r in lime_results if os.path.exists(r['image_path'])]

if len(test_images) < 3:
    # Fallback: sample fresh
    rng_local = np.random.RandomState(SEED)
    for folder, class_name in FOLDER_TO_CLASS.items():
        class_dir = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(class_dir):
            continue
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        sel   = rng_local.choice(files, size=min(IMAGES_PER_CLASS, len(files)), replace=False)
        test_images += [os.path.join(class_dir, f) for f in sel]

print(f"\nEvaluating {len(test_images)} images with Grad-CAM…\n")


# ── MAIN EVALUATION LOOP ─────────────────────────────────────────────────────

all_gc_results = []

for img_path in test_images:
    print(f"  Processing: {os.path.basename(img_path)}")
    try:
        pil, tensor = load_image(img_path)
        tensor = tensor.to(device)

        # Predicted class
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            class_idx  = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, class_idx].item())

        print(f"    Prediction: {class_names[class_idx]} ({confidence:.3f})")

        # Grad-CAM superpixel importance
        segments, gc_importance, cam_224 = get_gradcam_importance(pil, class_idx)

        # Deletion
        del_scores, del_auc = run_deletion(pil, segments, gc_importance, class_idx)
        print(f"    Grad-CAM Deletion AUC: {del_auc:.4f}")

        # Insertion
        ins_scores, ins_auc = run_insertion(pil, segments, gc_importance, class_idx)
        print(f"    Grad-CAM Insertion AUC: {ins_auc:.4f}")

        all_gc_results.append({
            'image_path':       img_path,
            'predicted_class':  class_names[class_idx],
            'gradcam_del_auc':  del_auc,
            'gradcam_ins_auc':  ins_auc,
            'del_scores':       del_scores,
            'ins_scores':       ins_scores,
        })
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        continue


# ── MERGE WITH LIME RESULTS ───────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("GRAD-CAM vs LIME COMPARISON")
print("=" * 65)

comparison_rows = []
for gc, lime in zip(all_gc_results, lime_results[:len(all_gc_results)]):
    row = {
        'image': os.path.basename(gc['image_path']),
        'class': gc['predicted_class'],
        'lime_del':  lime.get('lime_deletion_auc', 0),
        'gcam_del':  gc['gradcam_del_auc'],
        'lime_ins':  lime.get('lime_insertion_auc', 0),
        'gcam_ins':  gc['gradcam_ins_auc'],
    }
    comparison_rows.append(row)
    print(f"  {row['image']:<22} {row['class']:<18} "
          f"Del: LIME={row['lime_del']:.4f} GradCAM={row['gcam_del']:.4f} | "
          f"Ins: LIME={row['lime_ins']:.4f} GradCAM={row['gcam_ins']:.4f}")

avg_lime_del = np.mean([r['lime_del'] for r in comparison_rows])
avg_gcam_del = np.mean([r['gcam_del'] for r in comparison_rows])
avg_lime_ins = np.mean([r['lime_ins'] for r in comparison_rows])
avg_gcam_ins = np.mean([r['gcam_ins'] for r in comparison_rows])

print(f"\n  AVERAGE: Del: LIME={avg_lime_del:.4f} GradCAM={avg_gcam_del:.4f} | "
      f"Ins: LIME={avg_lime_ins:.4f} GradCAM={avg_gcam_ins:.4f}")


# ── COMPARISON BAR CHART ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fidelity Comparison: LIME vs Grad-CAM', fontsize=14, fontweight='bold')

x = np.arange(len(comparison_rows))
w = 0.35
names = [r['image'][:14] + '..' for r in comparison_rows]

for ax, key_lime, key_gcam, title, ylabel in [
    (axes[0], 'lime_del', 'gcam_del', 'Deletion AUC (Lower=Better)', 'AUC'),
    (axes[1], 'lime_ins', 'gcam_ins', 'Insertion AUC (Higher=Better)', 'AUC'),
]:
    vals_lime = [r[key_lime] for r in comparison_rows]
    vals_gcam = [r[key_gcam] for r in comparison_rows]
    ax.bar(x - w/2, vals_lime, w, label='LIME',     color='steelblue',  alpha=0.85)
    ax.bar(x + w/2, vals_gcam, w, label='Grad-CAM', color='darkorange', alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved: {OUTPUT_FIG}")


# ── SAVE ─────────────────────────────────────────────────────────────────────

output = {
    'summary': {
        'avg_lime_deletion_auc':  avg_lime_del,
        'avg_gcam_deletion_auc':  avg_gcam_del,
        'avg_lime_insertion_auc': avg_lime_ins,
        'avg_gcam_insertion_auc': avg_gcam_ins,
        'winner_deletion':  'LIME' if avg_lime_del < avg_gcam_del else 'Grad-CAM',
        'winner_insertion': 'LIME' if avg_lime_ins > avg_gcam_ins else 'Grad-CAM',
    },
    'per_image': comparison_rows
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)
print(f"✓ Results saved: {OUTPUT_JSON}")

gcam.remove_hooks()

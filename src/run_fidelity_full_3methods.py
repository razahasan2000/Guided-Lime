"""
run_fidelity_full_3methods.py
================================
Full 550-image fidelity evaluation computing THREE methods:
  - Standard LIME (no Grad-CAM gating)
  - Grad-CAM (superpixel ranking by mean Grad-CAM intensity)
  - Guided-LIME (multiplicative fusion: S(s) = L(s) * (1 + G(s)))
Plus a Random Baseline (5 shuffles of superpixel ordering).

Optimized: the LIME 1000-sample explanation is computed ONCE per image and
reused for both LIME and Guided-LIME scores (Guided-LIME reweights LIME
by Grad-CAM gating). This keeps total time ~2x LIME-only instead of 3x.
"""
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from lime.lime_image import LimeImageExplainer
import warnings
warnings.filterwarnings('ignore')

from gradcam_utils import SwinGradCAM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Load Config ---
with open('dataset_config.json', 'r') as f:
    config = json.load(f)
NUM_CLASSES = config['num_classes']
CLASS_MAPPING = config['class_to_idx']
idx_to_class = {v: k for k, v in CLASS_MAPPING.items()}
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

# --- Load Test Split ---
with open('test_split.json', 'r') as f:
    splits = json.load(f)
FOLDER_TO_CLASS = {
    'No_DR': 'No_DR', 'Mild': 'Mild', 'Moderate': 'Moderate',
    'Severe': 'Severe', 'Proliferate_DR': 'Proliferative_DR'
}
data_dir = 'data/colored_images'
test_images = []
test_class_map = {}
for folder, files in splits['test_set'].items():
    cls_name = FOLDER_TO_CLASS[folder]
    for fname in files:
        fpath = os.path.join(data_dir, folder, fname)
        test_images.append(fpath)
        test_class_map[fpath] = cls_name
print(f"[INFO] Test set: {len(test_images)} images")
for folder, files in splits['test_set'].items():
    print(f"  {FOLDER_TO_CLASS[folder]}: {len(files)}")

# --- Load Model ---
model = models.swin_t(weights=None)
in_features = model.head.in_features
model.head = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load('models/swin_transformer_final_generalized.pth', map_location=device))
model.eval().to(device)
print(f"[INFO] Model loaded")

# --- Grad-CAM hook ---
gcam = SwinGradCAM(model, device)

# --- Transforms ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Batched inference ---
def batched_predict(images_np, batch_size=128):
    transform = get_transform()
    probs_all = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i+batch_size]
        tensors = [transform(Image.fromarray(img.astype('uint8'))) for img in batch]
        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch_t)
            probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)

def predict_fn(images):
    return batched_predict(images, batch_size=128)

# --- LIME ---
def generate_lime_explanation(image_pil, num_samples=1000):
    image_np = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=SEED)
    def seg_fn(image):
        return slic(image, n_segments=100, compactness=10, sigma=1, start_label=1)
    explanation = explainer.explain_instance(
        image_np, classifier_fn=predict_fn, top_labels=1,
        hide_color=0, num_samples=num_samples, batch_size=128,
        segmentation_fn=seg_fn
    )
    predicted_class = explanation.top_labels[0]
    return explanation.segments, dict(explanation.local_exp[predicted_class]), predicted_class

# --- Grad-CAM scores per superpixel ---
def gradcam_per_superpixel(cam_224, segments):
    """Mean Grad-CAM intensity within each superpixel."""
    out = {}
    for sid in np.unique(segments):
        out[int(sid)] = float(cam_224[segments == sid].mean())
    return out

def normalize_scores(scores_dict):
    """Min-max normalize to [0, 1]. If constant, return 0.5."""
    vals = np.array(list(scores_dict.values()), dtype=float)
    if len(vals) == 0:
        return {}
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in scores_dict}
    return {k: float((v - mn) / (mx - mn)) for k, v in scores_dict.items()}

# --- Deletion / Insertion ---
def compute_deletion_curve(image_pil, segments, importance_order, pred_class, num_steps=20):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    total = len(importance_order)
    per_step = max(1, total // num_steps)
    mean_value = np.array([123.675, 116.28, 103.53])  # ImageNet mean (BGR)
    gray = np.full_like(image_np, mean_value)
    scores = []
    modified = image_np.copy()
    pil_img = Image.fromarray(modified.astype('uint8'))
    tensor = get_transform()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        scores.append(torch.softmax(out, 1)[0, pred_class].item())
    for step in range(num_steps):
        s = step * per_step
        e = min(s + per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            mask = segments == sid
            modified[mask] = gray[mask]
        pil_img = Image.fromarray(modified.astype('uint8'))
        tensor = get_transform()(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            scores.append(torch.softmax(out, 1)[0, pred_class].item())
        if e >= total:
            break
    x = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return float(auc)

def compute_insertion_curve(image_pil, segments, importance_order, pred_class, num_steps=20):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    total = len(importance_order)
    per_step = max(1, total // num_steps)
    mean_value = np.array([123.675, 116.28, 103.53])
    gray = np.full_like(image_np, mean_value)
    scores = []
    modified = gray.copy()
    pil_img = Image.fromarray(modified.astype('uint8'))
    tensor = get_transform()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        scores.append(torch.softmax(out, 1)[0, pred_class].item())
    for step in range(num_steps):
        s = step * per_step
        e = min(s + per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            mask = segments == sid
            modified[mask] = image_np[mask]
        pil_img = Image.fromarray(modified.astype('uint8'))
        tensor = get_transform()(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            scores.append(torch.softmax(out, 1)[0, pred_class].item())
        if e >= total:
            break
    x = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return float(auc)

# --- Random baseline ---
def random_baseline_curves(image_pil, segments, pred_class, num_steps=20, num_runs=5):
    unique_segs = np.unique(segments).tolist()
    del_aucs, ins_aucs = [], []
    for r in range(num_runs):
        rng = np.random.RandomState(SEED + r)
        shuffled = list(unique_segs)
        rng.shuffle(shuffled)
        del_auc = compute_deletion_curve(image_pil, segments, shuffled, pred_class, num_steps)
        ins_auc = compute_insertion_curve(image_pil, segments, shuffled, pred_class, num_steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
    return float(np.mean(del_aucs)), float(np.mean(ins_aucs))

# --- Main evaluation loop ---
OUT_PATH = 'fidelity_full_results.json'
PROGRESS_PATH = 'fidelity_full_progress.json'

done = set()
results = []
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH, 'r') as f:
        progress = json.load(f)
        done = set(progress.get('done', []))
        results = progress.get('results', [])
    print(f"[INFO] Resuming: {len(done)} already done")

t_start = time.time()
for idx, img_path in enumerate(test_images):
    if img_path in done:
        continue
    fname = os.path.basename(img_path)
    print(f"\n[{idx+1}/{len(test_images)}] {fname} ({test_class_map[img_path]})")
    t0 = time.time()

    try:
        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = get_transform()(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image_tensor)
            probs = torch.softmax(out, 1)
            pred_class = int(probs.argmax(1).item())
            pred_conf = float(probs[0, pred_class].item())

        # Step 1: LIME (raw surrogate coefficients)
        segs, lime_raw, _ = generate_lime_explanation(image_pil, num_samples=1000)

        # Step 2: Grad-CAM (one forward+backward)
        cam_224, _, _ = gcam.generate(image_tensor, class_idx=pred_class)
        gcam_raw = gradcam_per_superpixel(cam_224, segs)

        # Step 3: Compute the three importance rankings
        # (a) LIME ranking: sort by raw LIME weight descending
        lime_sorted = [s for s, _ in sorted(lime_raw.items(), key=lambda x: x[1], reverse=True)]
        # (b) Grad-CAM ranking: sort by normalized CAM intensity descending
        gcam_norm = normalize_scores(gcam_raw)
        gcam_sorted = [s for s, _ in sorted(gcam_norm.items(), key=lambda x: x[1], reverse=True)]
        # (c) Guided-LIME ranking: S(s) = L(s) * (1 + G(s))
        #     L is raw (signed), G is normalized to [0, 1]; (1 + G) in [1, 2].
        hybrid_score = {sid: lime_raw.get(sid, 0.0) * (1.0 + gcam_norm.get(sid, 0.0))
                        for sid in set(list(lime_raw.keys()) + list(gcam_norm.keys()))}
        guided_sorted = [s for s, _ in sorted(hybrid_score.items(), key=lambda x: x[1], reverse=True)]

        # Step 4: Deletion/Insertion for each method
        lime_del = compute_deletion_curve(image_pil, segs, lime_sorted, pred_class)
        lime_ins = compute_insertion_curve(image_pil, segs, lime_sorted, pred_class)
        gcam_del = compute_deletion_curve(image_pil, segs, gcam_sorted, pred_class)
        gcam_ins = compute_insertion_curve(image_pil, segs, gcam_sorted, pred_class)
        guided_del = compute_deletion_curve(image_pil, segs, guided_sorted, pred_class)
        guided_ins = compute_insertion_curve(image_pil, segs, guided_sorted, pred_class)

        # Step 5: Random baseline
        rand_del, rand_ins = random_baseline_curves(image_pil, segs, pred_class, num_runs=5)

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (idx + 1)) * (len(test_images) - idx - 1)
        print(f"  pred={idx_to_class[pred_class]} ({pred_conf:.3f})")
        print(f"  LIME   del/ins = {lime_del:.4f}/{lime_ins:.4f}")
        print(f"  GradCAM del/ins = {gcam_del:.4f}/{gcam_ins:.4f}")
        print(f"  Guided del/ins = {guided_del:.4f}/{guided_ins:.4f}")
        print(f"  Random del/ins = {rand_del:.4f}/{rand_ins:.4f}")
        print(f"  time={elapsed:.1f}s | ETA {eta/60:.1f}min")

        results.append({
            'image_path': img_path,
            'image_name': fname,
            'true_class': test_class_map[img_path],
            'predicted_class': idx_to_class[pred_class],
            'predicted_confidence': pred_conf,
            'lime_deletion_auc': lime_del,
            'lime_insertion_auc': lime_ins,
            'gradcam_deletion_auc': gcam_del,
            'gradcam_insertion_auc': gcam_ins,
            'guided_deletion_auc': guided_del,
            'guided_insertion_auc': guided_ins,
            'random_deletion_auc': rand_del,
            'random_insertion_auc': rand_ins,
        })
        done.add(img_path)
        with open(PROGRESS_PATH, 'w') as f:
            json.dump({'done': list(done), 'results': results}, f, indent=2)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"  ERROR: {e}")
        # Reset CUDA state on error to allow continuation
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        import traceback; traceback.print_exc()
        continue

# --- Aggregate ---
print(f"\n{'='*70}\nAGGREGATE FIDELITY RESULTS (n={len(results)})\n{'='*70}")
arr = lambda key: np.array([r[key] for r in results])

lime_del = arr('lime_deletion_auc')
gcam_del = arr('gradcam_deletion_auc')
guided_del = arr('guided_deletion_auc')
rand_del = arr('random_deletion_auc')
lime_ins = arr('lime_insertion_auc')
gcam_ins = arr('gradcam_insertion_auc')
guided_ins = arr('guided_insertion_auc')
rand_ins = arr('random_insertion_auc')

print(f"LIME    Deletion AUC: {lime_del.mean():.4f} +/- {lime_del.std():.4f}")
print(f"GradCAM Deletion AUC: {gcam_del.mean():.4f} +/- {gcam_del.std():.4f}")
print(f"Guided  Deletion AUC: {guided_del.mean():.4f} +/- {guided_del.std():.4f}")
print(f"Random  Deletion AUC: {rand_del.mean():.4f} +/- {rand_del.std():.4f}")
print()
print(f"LIME    Insertion AUC: {lime_ins.mean():.4f} +/- {lime_ins.std():.4f}")
print(f"GradCAM Insertion AUC: {gcam_ins.mean():.4f} +/- {gcam_ins.std():.4f}")
print(f"Guided  Insertion AUC: {guided_ins.mean():.4f} +/- {guided_ins.std():.4f}")
print(f"Random  Insertion AUC: {rand_ins.mean():.4f} +/- {rand_ins.std():.4f}")

# Wilcoxon tests
from scipy.stats import wilcoxon
print("\nWILCOXON SIGNED-RANK TESTS (vs Random):")
for name, arr_d, arr_i in [
    ('LIME', lime_del, lime_ins),
    ('GradCAM', gcam_del, gcam_ins),
    ('Guided-LIME', guided_del, guided_ins),
]:
    w, p = wilcoxon(arr_d, rand_del, alternative='less')
    print(f"  {name} Deletion < Random: p = {p:.4e}")
    w, p = wilcoxon(arr_i, rand_ins, alternative='greater')
    print(f"  {name} Insertion > Random: p = {p:.4e}")

# Wilcoxon: Guided-LIME vs LIME
w, p = wilcoxon(guided_del, lime_del, alternative='less')
print(f"\nGuided-LIME Deletion < LIME: p = {p:.4e}")
w, p = wilcoxon(guided_ins, lime_ins, alternative='greater')
print(f"Guided-LIME Insertion > LIME: p = {p:.4e}")

# Wilcoxon: Guided-LIME vs Grad-CAM
w, p = wilcoxon(guided_del, gcam_del, alternative='less')
print(f"Guided-LIME Deletion < Grad-CAM: p = {p:.4e}")
w, p = wilcoxon(guided_ins, gcam_ins, alternative='greater')
print(f"Guided-LIME Insertion > Grad-CAM: p = {p:.4e}")

# Per-class breakdown
from collections import defaultdict
by_class = defaultdict(list)
for r in results:
    by_class[r['true_class']].append(r)

per_class_stats = {}
for cls, items in by_class.items():
    lime_d = np.array([x['lime_deletion_auc'] for x in items])
    gcam_d = np.array([x['gradcam_deletion_auc'] for x in items])
    guid_d = np.array([x['guided_deletion_auc'] for x in items])
    rand_d = np.array([x['random_deletion_auc'] for x in items])
    lime_i = np.array([x['lime_insertion_auc'] for x in items])
    gcam_i = np.array([x['gradcam_insertion_auc'] for x in items])
    guid_i = np.array([x['guided_insertion_auc'] for x in items])
    rand_i = np.array([x['random_insertion_auc'] for x in items])
    per_class_stats[cls] = {
        'n': len(items),
        'lime_deletion_mean': float(lime_d.mean()), 'lime_deletion_std': float(lime_d.std()),
        'gradcam_deletion_mean': float(gcam_d.mean()), 'gradcam_deletion_std': float(gcam_d.std()),
        'guided_deletion_mean': float(guid_d.mean()), 'guided_deletion_std': float(guid_d.std()),
        'random_deletion_mean': float(rand_d.mean()), 'random_deletion_std': float(rand_d.std()),
        'lime_insertion_mean': float(lime_i.mean()), 'lime_insertion_std': float(lime_i.std()),
        'gradcam_insertion_mean': float(gcam_i.mean()), 'gradcam_insertion_std': float(gcam_i.std()),
        'guided_insertion_mean': float(guid_i.mean()), 'guided_insertion_std': float(guid_i.std()),
        'random_insertion_mean': float(rand_i.mean()), 'random_insertion_std': float(rand_i.std()),
    }
    print(f"  {cls} (n={len(items)}): LIME_d={lime_d.mean():.4f} GradCAM_d={gcam_d.mean():.4f} Guided_d={guid_d.mean():.4f} Random_d={rand_d.mean():.4f}")

# Save final aggregate
summary = {
    'n_images': len(results),
    'lime_deletion_mean': float(lime_del.mean()), 'lime_deletion_std': float(lime_del.std()),
    'gradcam_deletion_mean': float(gcam_del.mean()), 'gradcam_deletion_std': float(gcam_del.std()),
    'guided_deletion_mean': float(guided_del.mean()), 'guided_deletion_std': float(guided_del.std()),
    'random_deletion_mean': float(rand_del.mean()), 'random_deletion_std': float(rand_del.std()),
    'lime_insertion_mean': float(lime_ins.mean()), 'lime_insertion_std': float(lime_ins.std()),
    'gradcam_insertion_mean': float(gcam_ins.mean()), 'gradcam_insertion_std': float(gcam_ins.std()),
    'guided_insertion_mean': float(guided_ins.mean()), 'guided_insertion_std': float(guided_ins.std()),
    'random_insertion_mean': float(rand_ins.mean()), 'random_insertion_std': float(rand_ins.std()),
    'per_class': per_class_stats,
}
with open(OUT_PATH, 'w') as f:
    json.dump({'summary': summary, 'individual_results': results}, f, indent=2)
print(f"\n[INFO] Saved to {OUT_PATH}")
gcam.remove_hooks()
print("DONE")

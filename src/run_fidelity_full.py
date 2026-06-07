# ==============================================================================
# run_fidelity_full.py
# Full 550-image fidelity evaluation. Optimized for GPU.
# - Uses deterministic test split (test_split.json)
# - 1000 LIME perturbation samples (paper config)
# - 1 random baseline run per image (5 shuffles within)
# - No per-image plots (aggregate only)
# - Saves results incrementally for crash recovery
# ==============================================================================
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from lime.lime_image import LimeImageExplainer
import warnings
warnings.filterwarnings('ignore')

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

# --- Transforms ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

# --- Deletion / Insertion ---
def compute_deletion_curve(image_pil, segments, importance_order, pred_class, num_steps=20):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    total = len(importance_order)
    per_step = max(1, total // num_steps)
    mean_value = np.array([123.675, 116.28, 103.53])
    gray = np.full_like(image_np, mean_value)
    scores = []
    modified = image_np.copy()
    pil_img = Image.fromarray(modified.astype('uint8'))
    tensor = get_transform()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        scores.append(torch.softmax(out, 1)[0, pred_class].item())
    deleted = set()
    for step in range(num_steps):
        s = step * per_step
        e = min(s + per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            deleted.add(sid)
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
    return scores, auc

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
    inserted = set()
    for step in range(num_steps):
        s = step * per_step
        e = min(s + per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            inserted.add(sid)
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
    return scores, auc

# --- Random baseline: use 5 different shuffles of segments (no extra LIME) ---
def random_baseline_curves(image_pil, segments, pred_class, num_steps=20, num_runs=5):
    unique_segs = np.unique(segments).tolist()
    del_aucs, ins_aucs = [], []
    for r in range(num_runs):
        rng = np.random.RandomState(SEED + r)
        shuffled = list(unique_segs)
        rng.shuffle(shuffled)
        _, del_auc = compute_deletion_curve(image_pil, segments, shuffled, pred_class, num_steps)
        _, ins_auc = compute_insertion_curve(image_pil, segments, shuffled, pred_class, num_steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
    return float(np.mean(del_aucs)), float(np.mean(ins_aucs))

# --- Main evaluation loop ---
OUT_PATH = 'fidelity_full_results.json'
PROGRESS_PATH = 'fidelity_full_progress.json'

# Load progress if exists
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

        segs, importance, _ = generate_lime_explanation(image_pil, num_samples=1000)
        lime_sorted = [s for s, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)]
        _, lime_del = compute_deletion_curve(image_pil, segs, lime_sorted, pred_class, num_steps=20)
        _, lime_ins = compute_insertion_curve(image_pil, segs, lime_sorted, pred_class, num_steps=20)
        rand_del, rand_ins = random_baseline_curves(image_pil, segs, pred_class, num_steps=20, num_runs=5)

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (idx + 1)) * (len(test_images) - idx - 1)
        print(f"  pred={idx_to_class[pred_class]} ({pred_conf:.3f}) | LIME del/ins={lime_del:.4f}/{lime_ins:.4f} | Random del/ins={rand_del:.4f}/{rand_ins:.4f} | {elapsed:.1f}s | ETA {eta/60:.1f}min")

        results.append({
            'image_path': img_path,
            'image_name': fname,
            'true_class': test_class_map[img_path],
            'predicted_class': idx_to_class[pred_class],
            'predicted_confidence': pred_conf,
            'lime_deletion_auc': lime_del,
            'lime_insertion_auc': lime_ins,
            'random_deletion_auc': rand_del,
            'random_insertion_auc': rand_ins,
        })
        done.add(img_path)
        # Save progress after each image
        with open(PROGRESS_PATH, 'w') as f:
            json.dump({'done': list(done), 'results': results}, f, indent=2)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

# --- Aggregate ---
print(f"\n{'='*70}\nAGGREGATE FIDELITY RESULTS (n={len(results)})\n{'='*70}")
lime_del = np.array([r['lime_deletion_auc'] for r in results])
rand_del = np.array([r['random_deletion_auc'] for r in results])
lime_ins = np.array([r['lime_insertion_auc'] for r in results])
rand_ins = np.array([r['random_insertion_auc'] for r in results])
summary = {
    'n_images': len(results),
    'lime_deletion_mean': float(lime_del.mean()),
    'lime_deletion_std': float(lime_del.std()),
    'random_deletion_mean': float(rand_del.mean()),
    'random_deletion_std': float(rand_del.std()),
    'lime_insertion_mean': float(lime_ins.mean()),
    'lime_insertion_std': float(lime_ins.std()),
    'random_insertion_mean': float(rand_ins.mean()),
    'random_insertion_std': float(rand_ins.std()),
    'per_class': {}
}
for cls in ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']:
    cls_results = [r for r in results if r['true_class'] == cls]
    if cls_results:
        ld = np.array([r['lime_deletion_auc'] for r in cls_results])
        rd = np.array([r['random_deletion_auc'] for r in cls_results])
        li = np.array([r['lime_insertion_auc'] for r in cls_results])
        ri = np.array([r['random_insertion_auc'] for r in cls_results])
        summary['per_class'][cls] = {
            'n': len(cls_results),
            'lime_deletion_mean': float(ld.mean()),
            'lime_deletion_std': float(ld.std()),
            'random_deletion_mean': float(rd.mean()),
            'random_deletion_std': float(rd.std()),
            'lime_insertion_mean': float(li.mean()),
            'lime_insertion_std': float(li.std()),
            'random_insertion_mean': float(ri.mean()),
            'random_insertion_std': float(ri.std()),
        }
print(f"LIME Deletion AUC:   {summary['lime_deletion_mean']:.4f} +/- {summary['lime_deletion_std']:.4f}")
print(f"Random Deletion AUC: {summary['random_deletion_mean']:.4f} +/- {summary['random_deletion_std']:.4f}")
print(f"LIME Insertion AUC:   {summary['lime_insertion_mean']:.4f} +/- {summary['lime_insertion_std']:.4f}")
print(f"Random Insertion AUC: {summary['random_insertion_mean']:.4f} +/- {summary['random_insertion_std']:.4f}")

# Wilcoxon test
from scipy.stats import wilcoxon
if len(results) >= 10:
    w_stat, w_p_del = wilcoxon(lime_del, rand_del, alternative='less')
    w_stat2, w_p_ins = wilcoxon(lime_ins, rand_ins, alternative='greater')
    from math import sqrt
    z_del = w_stat
    r_del = abs(z_del) / sqrt(len(results))
    print(f"\nWilcoxon signed-rank test:")
    print(f"  Deletion (LIME < Random):  W={w_stat:.2f}, p={w_p_del:.2e}, Z={z_del:.2f}, r={r_del:.3f}")
    print(f"  Insertion (LIME > Random): W={w_stat2:.2f}, p={w_p_ins:.2e}")
    summary['wilcoxon_deletion'] = {'W': float(w_stat), 'p': float(w_p_del), 'Z': float(z_del), 'r': float(r_del)}
    summary['wilcoxon_insertion'] = {'W': float(w_stat2), 'p': float(w_p_ins)}

# Per-class print
print("\nPer-class results:")
for cls, d in summary['per_class'].items():
    print(f"  {cls:<20} n={d['n']:>4} | Del LIME={d['lime_deletion_mean']:.4f}±{d['lime_deletion_std']:.4f} Rand={d['random_deletion_mean']:.4f}±{d['random_deletion_std']:.4f} | Ins LIME={d['lime_insertion_mean']:.4f}±{d['lime_insertion_std']:.4f} Rand={d['random_insertion_mean']:.4f}±{d['random_insertion_std']:.4f}")

# Save final
with open(OUT_PATH, 'w') as f:
    json.dump({'summary': summary, 'individual_results': results}, f, indent=2)
print(f"\n[INFO] Saved to {OUT_PATH}")

# Cleanup progress
if os.path.exists(PROGRESS_PATH):
    os.remove(PROGRESS_PATH)
print("DONE")

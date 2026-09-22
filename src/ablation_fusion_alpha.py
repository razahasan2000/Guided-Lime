"""
ablation_fusion_alpha.py
==========================
Fusion parameter sensitivity: sweep alpha in S(s) = L(s) * (1 + alpha * G(s))
for alpha in [0, 0.5, 1, 2, 5, 10].

alpha=0 recovers Standard LIME, alpha=1 is the proposed Guided-LIME.
Uses existing 550-image test set.
"""

import os, json, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from lime.lime_image import LimeImageExplainer

from gradcam_utils import SwinGradCAM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# --- Load Config ---
with open('dataset_config.json', 'r') as f:
    config = json.load(f)
NUM_CLASSES = config['num_classes']
CLASS_MAPPING = config['class_to_idx']
idx_to_class = {v: k for k, v in CLASS_MAPPING.items()}

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

# --- Load Model ---
model = models.swin_t(weights=None)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('models/swin_transformer_final_generalized.pth', map_location=device))
model.eval().to(device)
print("[INFO] Model loaded")

# --- Grad-CAM ---
gcam = SwinGradCAM(model, device)

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

def gradcam_per_superpixel(cam_224, segments):
    out = {}
    for sid in np.unique(segments):
        out[int(sid)] = float(cam_224[segments == sid].mean())
    return out

def normalize_scores(scores_dict):
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
    mean_value = np.array([123.675, 116.28, 103.53])
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

# --- Alpha values to sweep ---
ALPHAS = [0, 0.5, 1, 2, 5, 10]

OUT_PATH = 'ablation_alpha_results.json'
PROGRESS_PATH = 'ablation_alpha_progress.json'

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

        # LIME explanation (shared across all alpha values)
        segs, lime_raw, _ = generate_lime_explanation(image_pil, num_samples=1000)

        # Grad-CAM (shared)
        cam_224, _, _ = gcam.generate(image_tensor, class_idx=pred_class)
        gcam_raw = gradcam_per_superpixel(cam_224, segs)
        gcam_norm = normalize_scores(gcam_raw)

        # For each alpha, compute fusion, run deletion/insertion
        per_alpha = {}
        for alpha in ALPHAS:
            hybrid_score = {
                sid: lime_raw.get(sid, 0.0) * (1.0 + alpha * gcam_norm.get(sid, 0.0))
                for sid in set(list(lime_raw.keys()) + list(gcam_norm.keys()))
            }
            sorted_segs = [s for s, _ in sorted(hybrid_score.items(), key=lambda x: x[1], reverse=True)]
            del_auc = compute_deletion_curve(image_pil, segs, sorted_segs, pred_class)
            ins_auc = compute_insertion_curve(image_pil, segs, sorted_segs, pred_class)
            per_alpha[str(alpha)] = {'deletion_auc': del_auc, 'insertion_auc': ins_auc}

        results.append({
            'image_path': img_path,
            'image_name': fname,
            'true_class': test_class_map[img_path],
            'predicted_class': test_class_map[img_path],
            'per_alpha': per_alpha,
        })
        done.add(img_path)

        with open(PROGRESS_PATH, 'w') as f:
            json.dump({'done': list(done), 'results': results}, f, indent=2)

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (idx + 1)) * (len(test_images) - idx - 1)
        print(f"  time={elapsed:.1f}s | ETA {eta/60:.1f}min")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        continue

# --- Aggregate ---
print(f"\n{'='*70}\nALPHA SWEEP RESULTS (n={len(results)})\n{'='*70}")
summary = {'n_images': len(results), 'per_alpha': {}}
for alpha in ALPHAS:
    dels = np.array([r['per_alpha'][str(alpha)]['deletion_auc'] for r in results])
    inss = np.array([r['per_alpha'][str(alpha)]['insertion_auc'] for r in results])
    summary['per_alpha'][str(alpha)] = {
        'deletion_mean': float(dels.mean()),
        'deletion_std': float(dels.std()),
        'insertion_mean': float(inss.mean()),
        'insertion_std': float(inss.std()),
    }
    print(f"alpha={alpha:5.1f}  Deletion AUC: {dels.mean():.4f} +/- {dels.std():.4f}  Insertion AUC: {inss.mean():.4f} +/- {inss.std():.4f}")

with open(OUT_PATH, 'w') as f:
    json.dump({'summary': summary, 'individual_results': results}, f, indent=2)
print(f"\n[INFO] Saved to {OUT_PATH}")

if os.path.exists(PROGRESS_PATH):
    os.remove(PROGRESS_PATH)

gcam.remove_hooks()
print("[INFO] Done")

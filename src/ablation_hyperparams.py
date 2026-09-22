"""
ablation_hyperparams.py
========================
Ablation on hyperparameters:
  1. Superpixel count: n in [25, 50, 100, 200, 500]
  2. Perturbation count: N in [100, 500, 1000, 2000, 5000]
  3. Baseline value: constant-gray vs. Gaussian blur vs. uniform noise

Runs on a 50-image class-balanced subset (10 per class) for speed.
"""

import os, json, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import slic
from skimage.filters import gaussian as skimage_gaussian
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

# Build per-class image lists
from collections import defaultdict
by_class_images = defaultdict(list)
for img_path in test_images:
    by_class_images[test_class_map[img_path]].append(img_path)

# Sample 10 per class (50 total)
subset_images = []
classes_sampled = []
rng = np.random.RandomState(SEED)
for cls_name in ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']:
    imgs = by_class_images[cls_name]
    chosen = list(rng.choice(imgs, size=min(10, len(imgs)), replace=False))
    subset_images.extend(chosen)
    for c in chosen:
        classes_sampled.append(cls_name)

print(f"[INFO] Ablation subset: {len(subset_images)} images (10 per class)")

# --- Load Model ---
model = models.swin_t(weights=None)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('models/swin_transformer_final_generalized.pth', map_location=device))
model.eval().to(device)
print("[INFO] Model loaded")

gcam = SwinGradCAM(model, device)

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

# --- Generic LIME wrapper ---
def generate_lime_explanation(image_pil, num_samples, n_segments):
    image_np = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=SEED)
    def seg_fn(image):
        return slic(image, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
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

def compute_deletion_curve(image_pil, segments, importance_order, pred_class, num_steps=20, baseline='gray'):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    total = len(importance_order)
    per_step = max(1, total // num_steps)

    if baseline == 'gray':
        mean_value = np.array([123.675, 116.28, 103.53])
        fill = np.full_like(image_np, mean_value)
    elif baseline == 'blur':
        fill = skimage_gaussian(image_np, sigma=5, channel_axis=-1).astype(np.float32)
    elif baseline == 'noise':
        fill = np.random.RandomState(SEED).randn(*image_np.shape).astype(np.float32) * 30 + 128
        fill = np.clip(fill, 0, 255).astype(np.float32)
    else:
        fill = np.full_like(image_np, 128.0)

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
            modified[mask] = fill[mask]
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

def compute_insertion_curve(image_pil, segments, importance_order, pred_class, num_steps=20, baseline='gray'):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    total = len(importance_order)
    per_step = max(1, total // num_steps)

    if baseline == 'gray':
        mean_value = np.array([123.675, 116.28, 103.53])
        fill = np.full_like(image_np, mean_value)
    elif baseline == 'blur':
        fill = skimage_gaussian(image_np, sigma=5, channel_axis=-1).astype(np.float32)
    elif baseline == 'noise':
        fill = np.random.RandomState(SEED).randn(*image_np.shape).astype(np.float32) * 30 + 128
        fill = np.clip(fill, 0, 255).astype(np.float32)
    else:
        fill = np.full_like(image_np, 128.0)

    scores = []
    modified = fill.copy()
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

# ==============================================================================
# Experiment configurations
# ==============================================================================

ABLATIONS = {
    'superpixel_count': {
        'label': 'Superpixel count',
        'variants': [25, 50, 100, 200, 500],
        'default_params': {'num_samples': 1000, 'n_segments': 100, 'baseline': 'gray'}
    },
    'perturbation_count': {
        'label': 'Perturbation count',
        'variants': [100, 500, 1000, 2000, 5000],
        'default_params': {'num_samples': 1000, 'n_segments': 100, 'baseline': 'gray'}
    },
    'baseline_type': {
        'label': 'Baseline type',
        'variants': ['gray', 'blur', 'noise'],
        'default_params': {'num_samples': 1000, 'n_segments': 100, 'baseline': 'gray'}
    },
}

results = {}
OUT_PATH = 'ablation_hyperparams_results.json'

for ab_name, ab_cfg in ABLATIONS.items():
    print(f"\n{'='*60}")
    print(f"Ablation: {ab_cfg['label']}")
    print(f"{'='*60}")

    ab_results = []
    for variant in ab_cfg['variants']:
        print(f"\n  Config: {ab_name} = {variant}")
        del_aucs, ins_aucs = [], []

        for img_idx, img_path in enumerate(subset_images):
            print(f"    [{img_idx+1}/{len(subset_images)}] {os.path.basename(img_path)}", end=' ')
            try:
                image_pil = Image.open(img_path).convert('RGB')
                image_tensor = get_transform()(image_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(image_tensor)
                    probs = torch.softmax(out, 1)
                    pred_class = int(probs.argmax(1).item())

                # Determine parameters based on ablation
                if ab_name == 'superpixel_count':
                    n_seg = variant
                    n_samp = 1000
                    bl = 'gray'
                elif ab_name == 'perturbation_count':
                    n_seg = 100
                    n_samp = variant
                    bl = 'gray'
                else:  # baseline_type
                    n_seg = 100
                    n_samp = 1000
                    bl = variant

                segs, lime_raw, _ = generate_lime_explanation(image_pil, num_samples=n_samp, n_segments=n_seg)

                cam_224, _, _ = gcam.generate(image_tensor, class_idx=pred_class)
                gcam_raw = gradcam_per_superpixel(cam_224, segs)
                gcam_norm = normalize_scores(gcam_raw)

                hybrid_score = {
                    sid: lime_raw.get(sid, 0.0) * (1.0 + gcam_norm.get(sid, 0.0))
                    for sid in set(list(lime_raw.keys()) + list(gcam_norm.keys()))
                }
                sorted_segs = [s for s, _ in sorted(hybrid_score.items(), key=lambda x: x[1], reverse=True)]

                del_auc = compute_deletion_curve(image_pil, segs, sorted_segs, pred_class, baseline=bl)
                ins_auc = compute_insertion_curve(image_pil, segs, sorted_segs, pred_class, baseline=bl)
                del_aucs.append(del_auc)
                ins_aucs.append(ins_auc)
                print(f"del={del_auc:.4f} ins={ins_auc:.4f}")

                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        ab_results.append({
            'variant': str(variant),
            'deletion_mean': float(np.mean(del_aucs)) if del_aucs else None,
            'deletion_std': float(np.std(del_aucs)) if del_aucs else None,
            'insertion_mean': float(np.mean(ins_aucs)) if ins_aucs else None,
            'insertion_std': float(np.std(ins_aucs)) if ins_aucs else None,
            'n_images': len(del_aucs),
        })
        print(f"  => Mean Del={np.mean(del_aucs):.4f} Ins={np.mean(ins_aucs):.4f} (n={len(del_aucs)})")

    results[ab_name] = ab_results

# Save
with open(OUT_PATH, 'w') as f:
    json.dump({'n_images': len(subset_images), 'ablation_results': results}, f, indent=2)
print(f"\n[INFO] Saved to {OUT_PATH}")

gcam.remove_hooks()
print("[INFO] Done")

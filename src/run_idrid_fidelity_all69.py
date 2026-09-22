#!/usr/bin/env python3
"""
run_idrid_fidelity_all69.py
===========================
Reviewer 1, Comment 2 (manuscript anujr_a-521, second round):
re-runs the exact fidelity protocol of Algorithm 3 on ALL 69 held-out IDRiD
test images (including the 18 misclassified ones) and reports Deletion /
Insertion AUC for BOTH:
  - n = 69  (all test images; directly comparable to the APTOS n = 550 protocol)
  - n = 51  (correctly classified only; the manuscript's primary analysis)

Methods: Standard LIME, Grad-CAM, Guided-LIME (alpha = 1.0), Random baseline.
The protocol matches run_fidelity_full_3methods.py of the Guided-Lime repo:
  * SLIC segmentation:  n_segments=100, compactness=10, sigma=1, start_label=1
  * LIME:               LimeImageExplainer, num_samples=1000, hide_color=0,
                        top_labels=1, random_state=42
  * Deletion/Insertion: 20 steps, constant baseline = ImageNet mean
                        (123.675, 116.28, 103.53)
  * Random baseline:    5 shuffles of superpixel ordering

Extend the METHOD_REGISTRY below with your RISE / SHAP / Integrated Gradients /
Attention-Rollout implementations (the same per-image interface) to reproduce
the full 7-method comparison on n = 69.

Usage:
  1. Place the IDRiD grading test images in a folder structure:
       idrid_test/No_DR/*.jpg  idrid_test/Mild/*.jpg  idrid_test/Moderate/*.jpg
       idrid_test/Severe/*.jpg idrid_test/Proliferate_DR/*.jpg
     (69 images in total: 20 No_DR, 3 Mild, 24 Moderate, 13 Severe, 9 Prolif.)
  2. Provide the IDRiD-fine-tuned Swin-T checkpoint (state_dict).
  3. pip install torch torchvision lime scikit-image scipy numpy pillow
  4. python run_idrid_fidelity_all69.py --images idrid_test \
         --checkpoint models/swin_t_idrid_finetuned.pth --out idrid69_results
  5. Copy the printed "READY-TO-PASTE" values into the two yellow placeholders
     in the Subset-sensitivity-analysis paragraph of the manuscript.

Runtime: ~35-50 min for 69 images on an RTX 4070 Laptop GPU (8 GB VRAM).
"""
import os
import json
import time
import argparse
import warnings

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

SEED = 42
NUM_SAMPLES = 1000          # LIME perturbation samples (K)
NUM_SEGMENTS = 100          # SLIC superpixels (n)
NUM_STEPS = 20              # deletion/insertion curve steps
NUM_RANDOM_RUNS = 5         # random-baseline shuffles
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])  # constant baseline

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------- model
def load_model(checkpoint_path, num_classes=5):
    model = models.swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SwinGradCAM:
    """Grad-CAM for Swin-T via forward/backward hooks on the last stage norm."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        target = model.norm  # final LayerNorm over Stage-4 tokens (torchvision Swin)
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach().reshape(out.shape[0], -1, out.shape[-1])

    def _bwd_hook(self, module, grad_in, grad_out):
        g = grad_out[0].detach()
        self.gradients = g.reshape(g.shape[0], -1, g.shape[-1])

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward()
        acts = self.activations[0]          # (tokens, C)
        grads = self.gradients[0]           # (tokens, C)
        weights = grads.mean(dim=0)         # GAP over tokens
        cam = (weights * acts).sum(dim=-1)  # (tokens,)
        n = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(n, n)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0].cpu().numpy()
        return cam, class_idx

    def remove_hooks(self):
        pass


# ----------------------------------------------------------------- core primitives
def batched_predict(model, images_np, batch_size=128):
    tf = get_transform()
    probs_all = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i + batch_size]
        tensors = [tf(Image.fromarray(img.astype('uint8'))) for img in batch]
        batch_t = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            logits = model(batch_t)
            probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def generate_lime(model, image_pil, num_samples=NUM_SAMPLES):
    image_np = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=SEED)

    def seg_fn(image):
        return slic(image, n_segments=NUM_SEGMENTS, compactness=10,
                    sigma=1, start_label=1)

    def predict_fn(images):
        return batched_predict(model, images, batch_size=128)

    explanation = explainer.explain_instance(
        image_np, classifier_fn=predict_fn, top_labels=1,
        hide_color=0, num_samples=num_samples, batch_size=128,
        segmentation_fn=seg_fn,
    )
    pred = explanation.top_labels[0]
    return explanation.segments, dict(explanation.local_exp[pred]), pred


def gradcam_per_superpixel(cam_224, segments):
    return {int(sid): float(cam_224[segments == sid].mean())
            for sid in np.unique(segments)}


def normalize_scores(scores_dict):
    vals = np.array(list(scores_dict.values()), dtype=float)
    if len(vals) == 0:
        return {}
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in scores_dict}
    return {k: float((v - mn) / (mx - mn)) for k, v in scores_dict.items()}


def compute_deletion_curve(image_pil, segments, importance_order, pred_class,
                           num_steps=NUM_STEPS):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    tf = get_transform()
    total = len(importance_order)
    per_step = max(1, total // num_steps)
    gray = np.full_like(image_np, IMAGENET_MEAN_255)
    scores = []
    modified = image_np.copy()
    tensor = tf(Image.fromarray(modified.astype('uint8'))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores.append(torch.softmax(model(tensor), 1)[0, pred_class].item())
    for step in range(num_steps):
        s, e = step * per_step, min((step + 1) * per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            modified[segments == sid] = gray[segments == sid]
        tensor = tf(Image.fromarray(modified.astype('uint8'))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            scores.append(torch.softmax(model(tensor), 1)[0, pred_class].item())
        if e >= total:
            break
    x = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return float(auc)


def compute_insertion_curve(image_pil, segments, importance_order, pred_class,
                            num_steps=NUM_STEPS):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    tf = get_transform()
    total = len(importance_order)
    per_step = max(1, total // num_steps)
    gray = np.full_like(image_np, IMAGENET_MEAN_255)
    scores = []
    modified = gray.copy()
    tensor = tf(Image.fromarray(modified.astype('uint8'))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores.append(torch.softmax(model(tensor), 1)[0, pred_class].item())
    for step in range(num_steps):
        s, e = step * per_step, min((step + 1) * per_step, total)
        for idx in range(s, e):
            sid = importance_order[idx]
            modified[segments == sid] = image_np[segments == sid]
        tensor = tf(Image.fromarray(modified.astype('uint8'))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            scores.append(torch.softmax(model(tensor), 1)[0, pred_class].item())
        if e >= total:
            break
    x = np.linspace(0, 1, len(scores))
    auc = np.trapezoid(scores, x) if hasattr(np, 'trapezoid') else np.trapz(scores, x)
    return float(auc)


def random_baseline(image_pil, segments, pred_class, num_runs=NUM_RANDOM_RUNS):
    unique_segs = np.unique(segments).tolist()
    del_aucs, ins_aucs = [], []
    for r in range(num_runs):
        rng = np.random.RandomState(SEED + r)
        order = list(unique_segs)
        rng.shuffle(order)
        del_aucs.append(compute_deletion_curve(image_pil, segments, order, pred_class))
        ins_aucs.append(compute_insertion_curve(image_pil, segments, order, pred_class))
    return float(np.mean(del_aucs)), float(np.mean(ins_aucs))


# ------------------------------------------------------------------- main pipeline
def evaluate_image(model, gcam, img_path, true_class):
    image_pil = Image.open(img_path).convert('RGB')
    tensor = get_transform()(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)
        pred = int(probs.argmax(1).item())

    segs, lime_raw, _ = generate_lime(model, image_pil)
    cam_224, _ = gcam.generate(tensor, class_idx=pred)
    gcam_norm = normalize_scores(gradcam_per_superpixel(cam_224, segs))

    lime_order = [s for s, _ in sorted(lime_raw.items(), key=lambda kv: kv[1], reverse=True)]
    gcam_order = [s for s, _ in sorted(gcam_norm.items(), key=lambda kv: kv[1], reverse=True)]
    hybrid = {sid: lime_raw.get(sid, 0.0) * (1.0 + gcam_norm.get(sid, 0.0))
              for sid in set(list(lime_raw.keys()) + list(gcam_norm.keys()))}
    guided_order = [s for s, _ in sorted(hybrid.items(), key=lambda kv: kv[1], reverse=True)]

    rec = {
        'image': os.path.basename(img_path), 'true_class': true_class,
        'predicted_class': int(pred),
        'correct': int(pred == true_class),
        'lime_deletion_auc': compute_deletion_curve(image_pil, segs, lime_order, pred),
        'lime_insertion_auc': compute_insertion_curve(image_pil, segs, lime_order, pred),
        'gradcam_deletion_auc': compute_deletion_curve(image_pil, segs, gcam_order, pred),
        'gradcam_insertion_auc': compute_insertion_curve(image_pil, segs, gcam_order, pred),
        'guided_deletion_auc': compute_deletion_curve(image_pil, segs, guided_order, pred),
        'guided_insertion_auc': compute_insertion_curve(image_pil, segs, guided_order, pred),
        **({}),
    }
    rd, ri_ = random_baseline(image_pil, segs, pred)
    rec['random_deletion_auc'] = rd
    rec['random_insertion_auc'] = ri_
    return rec


METHODS = {
    'Standard LIME': ('lime_deletion_auc', 'lime_insertion_auc'),
    'Grad-CAM': ('gradcam_deletion_auc', 'gradcam_insertion_auc'),
    'Guided-LIME': ('guided_deletion_auc', 'guided_insertion_auc'),
    'Random Baseline': ('random_deletion_auc', 'random_insertion_auc'),
    # Extend here with RISE / SHAP / IG / Attention once plugged in above.
}


def summarize(records):
    out = {}
    for name, (dk, ik) in METHODS.items():
        d = np.array([r_[dk] for r_ in records], dtype=float)
        i = np.array([r_[ik] for r_ in records], dtype=float)
        out[name] = {
            'deletion_mean': float(d.mean()), 'deletion_std': float(d.std()),
            'insertion_mean': float(i.mean()), 'insertion_std': float(i.std()),
        }
    return out


def paired_tests(records_a, records_b):
    """Wilcoxon: each method vs Random Baseline (deletion less, insertion greater)."""
    res = {}
    bmap = {r_['image']: r_ for r_ in records_b}

    def safe_p(a, b, alternative):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if a.size == 0 or np.allclose(a - b, 0.0):
            return float('nan')  # undefined for degenerate samples
        try:
            return float(wilcoxon(a, b, alternative=alternative)[1])
        except ValueError:
            return float('nan')

    for name, (dk, ik) in METHODS.items():
        if name == 'Random Baseline':
            continue
        a_del = np.array([r_[dk] for r_ in records_a])
        a_ins = np.array([r_[ik] for r_ in records_a])
        b_del = np.array([bmap[r_['image']]['random_deletion_auc'] for r_ in records_a])
        b_ins = np.array([bmap[r_['image']]['random_insertion_auc'] for r_ in records_a])
        res[f'{name} vs Random'] = {'deletion_p': safe_p(a_del, b_del, 'less'),
                                    'insertion_p': safe_p(a_ins, b_ins, 'greater')}
    g_del = np.array([r_['guided_deletion_auc'] for r_ in records_a])
    l_del = np.array([r_['lime_deletion_auc'] for r_ in records_a])
    res['Guided-LIME vs Standard LIME'] = {'deletion_p': safe_p(g_del, l_del, 'less')}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='idrid_test', help='folder with class subfolders')
    ap.add_argument('--checkpoint', default='models/swin_t_idrid_finetuned.pth')
    ap.add_argument('--out', default='idrid69_results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    global model
    model = load_model(args.checkpoint)
    gcam = SwinGradCAM(model, DEVICE)
    print(f'[INFO] device={DEVICE}; model loaded from {args.checkpoint}')

    CLASS_FOLDERS = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    class_to_idx = {c: i for i, c in enumerate(CLASS_FOLDERS)}
    images = []
    for c in CLASS_FOLDERS:
        d = os.path.join(args.images, c)
        if not os.path.isdir(d):
            raise FileNotFoundError(f'missing class folder: {d}')
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append((os.path.join(d, f), class_to_idx[c]))
    print(f'[INFO] found {len(images)} test images '
          f'(expected 69: 20 No_DR, 3 Mild, 24 Moderate, 13 Severe, 9 Prolif)')

    records = []
    t0 = time.time()
    for n, (path, y) in enumerate(images, 1):
        rec = evaluate_image(model, gcam, path, y)
        records.append(rec)
        if n % 10 == 0 or n == len(images):
            print(f'  [{n}/{len(images)}] elapsed {(time.time() - t0) / 60:.1f} min')
        with open(os.path.join(args.out, 'progress.json'), 'w') as f:
            json.dump(records, f, indent=2)

    correct = [r_ for r_ in records if r_['correct'] == 1]
    summary = {
        'n_all': len(records),
        'n_correct_only': len(correct),
        'summary_n69_all_images': summarize(records),
        'summary_n51_correct_only': summarize(correct),
        'wilcoxon_n69': paired_tests(records, records),
        'wilcoxon_n51': paired_tests(correct, correct),
        'per_record': records,
    }
    with open(os.path.join(args.out, 'idrid_fidelity_n69_and_n51.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 78)
    print('READY-TO-PASTE VALUES FOR THE MANUSCRIPT PLACEHOLDERS')
    print('=' * 78)
    print('\n-- n = 69 (all test images) --')
    for name, s in summary['summary_n69_all_images'].items():
        print(f"  {name:<16s} Deletion {s['deletion_mean']:.4f} +/- {s['deletion_std']:.3f}   "
              f"Insertion {s['insertion_mean']:.4f} +/- {s['insertion_std']:.3f}")
    print('\n-- n = 51 (correctly classified only; should match Table 9 trends) --')
    for name, s in summary['summary_n51_correct_only'].items():
        print(f"  {name:<16s} Deletion {s['deletion_mean']:.4f} +/- {s['deletion_std']:.3f}   "
              f"Insertion {s['insertion_mean']:.4f} +/- {s['insertion_std']:.3f}")
    print('\n-- Wilcoxon (n = 69) --')
    for k, v in summary['wilcoxon_n69'].items():
        print(f"  {k:<32s} {v}")
    print('\nManuscript sentence: "The aggregate Deletion/Insertion AUC under the '
          'n = 69 protocol are [Guided-LIME: <del> / <ins>; Standard LIME: <del> / <ins>], '
          'and the resulting method ranking is [ranking]."')
    print(f'\nSaved: {os.path.join(args.out, "idrid_fidelity_n69_and_n51.json")}')
    print('DONE')


if __name__ == '__main__':
    main()

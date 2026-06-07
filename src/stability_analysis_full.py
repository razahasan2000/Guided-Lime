# ==============================================================================
# stability_analysis_full.py
# Full 550-image stability evaluation. Optimized for GPU.
# - Uses deterministic test split (test_split.json)
# - 500 LIME samples (reduced for speed)
# - 1 noise + 1 rotation perturbation per image
# - Saves incrementally
# ==============================================================================
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

with open('dataset_config.json', 'r') as f:
    config = json.load(f)
NUM_CLASSES = config['num_classes']
CLASS_MAPPING = config['class_to_idx']
idx_to_class = {v: k for k, v in CLASS_MAPPING.items()}

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

model = models.swin_t(weights=None)
in_features = model.head.in_features
model.head = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load('models/swin_transformer_final_generalized.pth', map_location=device))
model.eval().to(device)

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

def generate_lime_explanation(image_pil, num_samples=500):
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

def apply_noise(image_pil, sigma=5, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    img_np = np.array(image_pil.resize((224, 224))).astype(np.float32)
    noise = rng.normal(0, sigma, img_np.shape)
    return Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))

def apply_rotation(image_pil, angle=2.0):
    return image_pil.resize((224, 224)).rotate(angle)

def jaccard_top_k(scores1, scores2, top_k=10):
    s1 = sorted(scores1.items(), key=lambda x: x[1], reverse=True)[:top_k]
    s2 = sorted(scores2.items(), key=lambda x: x[1], reverse=True)[:top_k]
    set1 = set(s[0] for s in s1)
    set2 = set(s[0] for s in s2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

OUT_PATH = 'stability_full_results.json'
PROGRESS_PATH = 'stability_full_progress.json'

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
        img_pil = Image.open(img_path).convert('RGB')
        segs_orig, scores_orig, pred_orig = generate_lime_explanation(img_pil, num_samples=500)

        stability_scores = []
        perturbations_detail = []

        # 1 noise perturbation
        for mode, fn, kwargs in [
            ('noise', apply_noise, {'sigma': 5, 'seed': SEED}),
            ('rotate', apply_rotation, {'angle': 2.0}),
        ]:
            try:
                perturbed = fn(img_pil, **kwargs)
                segs_p, scores_p, pred_p = generate_lime_explanation(perturbed, num_samples=500)
                if pred_p == pred_orig:
                    j = jaccard_top_k(scores_orig, scores_p, top_k=10)
                    stability_scores.append(j)
                    perturbations_detail.append({'mode': mode, 'jaccard': j})
            except Exception as e:
                print(f"  ! {mode} error: {e}")

        avg_stab = float(np.mean(stability_scores)) if stability_scores else 0.0
        std_stab = float(np.std(stability_scores)) if stability_scores else 0.0
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (idx + 1)) * (len(test_images) - idx - 1)
        print(f"  pred={idx_to_class[pred_orig]} | stability={avg_stab:.4f} +/- {std_stab:.4f} | {elapsed:.1f}s | ETA {eta/60:.1f}min")

        results.append({
            'image_path': img_path,
            'image_name': fname,
            'true_class': test_class_map[img_path],
            'predicted_class': idx_to_class[pred_orig],
            'stability': avg_stab,
            'std': std_stab,
            'n_perturbations': len(stability_scores),
            'perturbations': perturbations_detail,
        })
        done.add(img_path)
        with open(PROGRESS_PATH, 'w') as f:
            json.dump({'done': list(done), 'results': results}, f, indent=2)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

# Aggregate
print(f"\n{'='*70}\nAGGREGATE STABILITY RESULTS (n={len(results)})\n{'='*70}")
stab_scores = np.array([r['stability'] for r in results])
summary = {
    'n_images': len(results),
    'global_mean': float(stab_scores.mean()),
    'global_std': float(stab_scores.std()),
    'per_class': {}
}
for cls in ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']:
    cls_results = [r for r in results if r['true_class'] == cls]
    if cls_results:
        cls_s = np.array([r['stability'] for r in cls_results])
        summary['per_class'][cls] = {
            'n': len(cls_results),
            'mean': float(cls_s.mean()),
            'std': float(cls_s.std()),
        }
print(f"Global Stability: {summary['global_mean']:.4f} +/- {summary['global_std']:.4f}")
print("\nPer-class stability:")
for cls, d in summary['per_class'].items():
    print(f"  {cls:<20} n={d['n']:>4} | {d['mean']:.4f} +/- {d['std']:.4f}")

with open(OUT_PATH, 'w') as f:
    json.dump({'summary': summary, 'individual_results': results}, f, indent=2)
print(f"\n[INFO] Saved to {OUT_PATH}")
if os.path.exists(PROGRESS_PATH):
    os.remove(PROGRESS_PATH)
print("DONE")

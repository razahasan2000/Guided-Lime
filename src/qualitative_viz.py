"""
qualitative_viz.py
================================
Tier 2B: Qualitative Visualization Per DR Class

Generates a 3-panel figure for one representative image per class:
  Panel 1 – Original retinal fundus image
  Panel 2 – LIME superpixel overlay (green=positive, red=negative)
  Panel 3 – Grad-CAM heatmap overlay (jet, alpha=0.5)

Writes: qualitative_<ClassName>.png  (one per class, 5 total)
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import models
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import gaussian as skimage_gaussian
from lime.lime_image import LimeImageExplainer
import warnings
warnings.filterwarnings('ignore')

from gradcam_utils import SwinGradCAM, get_transform, load_image

# ── CONFIG ───────────────────────────────────────────────────────────────────
SEED            = 42
NUM_SEGMENTS    = 100
LIME_SAMPLES    = 500   # Lower count for quick qualitative viz
NUM_LIME_FEATS  = 10
MODEL_PATH      = 'models/swin_transformer_final_generalized.pth'
CONFIG_FILE     = 'dataset_config.json'
DATA_DIR        = 'data/colored_images'

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


def get_lime_mask(image_pil, class_idx, n_pos=5, n_neg=3):
    """Return (pos_mask, neg_mask) boolean arrays at 224x224."""
    img_np   = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=SEED)

    def seg_fn(img):
        return slic(img, n_segments=NUM_SEGMENTS, compactness=10, sigma=1, start_label=1)

    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=1,
        hide_color=0, num_samples=LIME_SAMPLES,
        batch_size=32, segmentation_fn=seg_fn
    )
    segments = explanation.segments
    scores   = dict(explanation.local_exp[class_idx])

    sorted_pos = sorted([k for k, v in scores.items() if v > 0],
                        key=lambda k: scores[k], reverse=True)[:n_pos]
    sorted_neg = sorted([k for k, v in scores.items() if v < 0],
                        key=lambda k: scores[k])[:n_neg]

    pos_mask = np.isin(segments, sorted_pos)
    neg_mask = np.isin(segments, sorted_neg)
    return pos_mask, neg_mask, segments


def overlay_lime(img_np, pos_mask, neg_mask):
    """Create RGBA overlay image with green=positive, red=negative."""
    overlay = img_np.astype(float).copy()
    alpha_layer = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    alpha_layer[pos_mask] = [0, 200, 80, 130]
    alpha_layer[neg_mask] = [220, 30, 30, 130]
    result = img_np.copy()
    for c in range(3):
        fg = alpha_layer[:, :, c].astype(float)
        a  = alpha_layer[:, :, 3].astype(float) / 255.0
        result[:, :, c] = (1 - a) * result[:, :, c] + a * fg
    return result.clip(0, 255).astype(np.uint8)


def overlay_gradcam(img_np, cam_224, alpha=0.5):
    """Overlay jet-colored Grad-CAM on image."""
    heatmap = cm.jet(cam_224)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    blended = (alpha * heatmap + (1 - alpha) * img_np).clip(0, 255).astype(np.uint8)
    return blended


# ── PER-CLASS VISUALIZATION ───────────────────────────────────────────────────

rng = np.random.RandomState(SEED)

for folder, class_name in FOLDER_TO_CLASS.items():
    class_dir = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(class_dir):
        print(f"  Warning: {class_dir} not found — skipping {class_name}")
        continue

    files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
    if len(files) == 0:
        continue

    img_file = rng.choice(files)
    img_path = os.path.join(class_dir, img_file)
    print(f"\n  [{class_name}] → {img_file}")

    pil, tensor = load_image(img_path)
    tensor      = tensor.to(device)
    img_np_224  = np.array(pil.resize((224, 224)))

    # Model prediction
    with torch.no_grad():
        logits    = model_nn(tensor)
        probs     = torch.softmax(logits, dim=1)
        class_idx = int(probs.argmax(dim=1).item())
        conf      = float(probs[0, class_idx].item())
    pred_name = class_names[class_idx]
    print(f"    Prediction: {pred_name} ({conf:.3f})")

    # LIME masks
    print("    Generating LIME explanation…")
    pos_mask, neg_mask, segs = get_lime_mask(pil, class_idx)

    # Grad-CAM
    print("    Generating Grad-CAM…")
    cam_224, _, _ = gcam.generate(tensor, class_idx=class_idx)

    # Build overlays
    lime_img  = overlay_lime(img_np_224, pos_mask, neg_mask)
    gcam_img  = overlay_gradcam(img_np_224, cam_224)

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Class: {class_name}  |  Prediction: {pred_name} ({conf:.1%})",
        fontsize=14, fontweight='bold'
    )

    axes[0].imshow(img_np_224);   axes[0].set_title('Original',        fontsize=12); axes[0].axis('off')
    axes[1].imshow(lime_img);     axes[1].set_title('LIME Explanation\n(Green=+, Red=−)', fontsize=12); axes[1].axis('off')
    axes[2].imshow(gcam_img);     axes[2].set_title('Grad-CAM Heatmap', fontsize=12); axes[2].axis('off')

    plt.tight_layout()
    save_path = f"qualitative_{class_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {save_path}")

gcam.remove_hooks()
print("\n✓ All qualitative figures generated.")

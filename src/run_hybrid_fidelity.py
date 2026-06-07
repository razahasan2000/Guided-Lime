import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from hybrid_explainer import GuidedLIME
from xai_utils import generate_lime_explanation, deletion_test, insertion_test, preprocess_image

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
model = models.swin_t(weights=None)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('models/swin_transformer_final_generalized.pth', map_location=device))
model.eval().to(device)

guided_lime = GuidedLIME(model, device)

def run_hybrid_comparison(image_path):
    print(f"Evaluating Hybrid vs Pure LIME for: {image_path}")
    img_pil, img_tensor, _ = preprocess_image(image_path)

    # Get Prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = torch.softmax(output, dim=1).argmax(dim=1).item()

    # 1. Pure LIME
    seg_l, scores_l, _ = generate_lime_explanation(img_pil, model, device)
    del_l, auc_l_del = deletion_test(img_pil, seg_l, scores_l, pred_class, model, device)
    ins_l, auc_l_ins = insertion_test(img_pil, seg_l, scores_l, pred_class, model, device)

    # 2. Hybrid LIME
    seg_h, scores_h, _ = guided_lime.generate_hybrid_explanation(img_pil)
    del_h, auc_h_del = deletion_test(img_pil, seg_h, scores_h, pred_class, model, device)
    ins_h, auc_h_ins = insertion_test(img_pil, seg_h, scores_h, pred_class, model, device)

    return {
        'image': image_path,
        'lime_del_auc': auc_l_del,
        'hybrid_del_auc': auc_h_del,
        'lime_ins_auc': auc_l_ins,
        'hybrid_ins_auc': auc_h_ins,
        'del_improvement': auc_l_del / auc_h_del if auc_h_del > 0 else 0,
        'ins_improvement': auc_h_ins / auc_l_ins if auc_l_ins > 0 else 0
    }

if __name__ == "__main__":
    # Simple test on 2 images
    data_dir = 'data/colored_images/Moderate'
    test_imgs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)[:2]]

    all_res = []
    for img in test_imgs:
        all_res.append(run_hybrid_comparison(img))

    print("\n--- HYBRID vs PURE LIME RESULTS ---")
    for r in all_res:
        print(f"Image: {os.path.basename(r['image'])}")
        print(f"  Deletion AUC: LIME {r['lime_del_auc']:.4f} -> Hybrid {r['hybrid_del_auc']:.4f}")
        print(f"  Insertion AUC: LIME {r['lime_ins_auc']:.4f} -> Hybrid {r['hybrid_ins_auc']:.4f}")

    with open('hybrid_comparison_results.json', 'w') as f:
        json.dump(all_res, f, indent=2)

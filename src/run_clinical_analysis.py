import os
import torch
import numpy as np
import json
from torchvision import models
import torch.nn as nn
from clinical_alignment import run_clinical_analysis

# ==============================================================================
# RUNNER SCRIPT
# ==============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the exact model used in your project
    NUM_CLASSES = 5
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    model_weights_path = 'models/swin_transformer_final_generalized.pth'
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f" Loaded model from {model_weights_path}")
    else:
        print(f"Error: Model weights not found at {model_weights_path}")
        return

    model.eval().to(device)

    # Define image samples to test
    data_dir = 'data/colored_images'
    classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    sampled_images = []

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.png')]
            sampled_images.extend(imgs[:2]) # Take 2 from each class

    print(f"Analyzing clinical alignment for {len(sampled_images)} images...")
    results = run_clinical_analysis(sampled_images, model, device)

    with open('clinical_alignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone. Results saved to clinical_alignment_results.json")

if __name__ == "__main__":
    main()

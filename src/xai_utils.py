import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.segmentation import slic
from skimage.filters import gaussian as skimage_gaussian
import random
from lime.lime_image import LimeImageExplainer

# ==============================================================================
# SHARED UTILITIES FOR XAI EVALUATION
# ==============================================================================

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path):
    """Load and preprocess an image for model inference."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor, original_size

def generate_lime_explanation(image_pil, model, device, num_samples=1000):
    """Generate LIME explanation for an image."""
    image_np = np.array(image_pil.resize((224, 224)))
    explainer = LimeImageExplainer(random_state=42)

    def predict_fn(images):
        transform = get_transform()
        batch_tensors = []
        for img in images:
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor = transform(pil_img)
            batch_tensors.append(tensor)
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def custom_segmentation(image):
        return slic(image, n_segments=100, compactness=10, sigma=1, start_label=1)

    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        batch_size=32,
        segmentation_fn=custom_segmentation
    )

    predicted_class = explanation.top_labels[0]
    segments = explanation.segments
    importance_scores = dict(explanation.local_exp[predicted_class])

    return segments, importance_scores, predicted_class

def deletion_test(image_pil, segments, importance_scores, predicted_class, model, device, num_steps=20):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    sorted_segments = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    segment_ids_ordered = [s[0] for s in sorted_segments]

    total_segments = len(segment_ids_ordered)
    segments_per_step = max(1, total_segments // num_steps)
    
    # Use constant gray value (mean pixel value) as deletion baseline
    # Calculate mean pixel value from training set (using ImageNet mean as approximation)
    mean_value = np.array([123.675, 116.28, 103.53])  # RGB mean values * 255 to match 0-255 scale
    # Convert to BGR since we're working with RGB images
    mean_value_bgr = mean_value[::-1]  # Reverse to get BGR order
    gray_image = np.full_like(image_np, mean_value_bgr)

    deletion_scores = []
    modified_image = image_np.copy()
    transform = get_transform()

    # Initial confidence
    pil_img = Image.fromarray(modified_image.astype('uint8'))
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        deletion_scores.append(probs[0, predicted_class].item())

    deleted_segments = set()
    for step in range(num_steps):
        start_idx = step * segments_per_step
        end_idx = min(start_idx + segments_per_step, total_segments)
        for idx in range(start_idx, end_idx):
            seg_id = segment_ids_ordered[idx]
            deleted_segments.add(seg_id)
            mask = segments == seg_id
            modified_image[mask] = gray_image[mask]

        pil_img = Image.fromarray(modified_image.astype('uint8'))
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            deletion_scores.append(probs[0, predicted_class].item())
        if end_idx >= total_segments: break

    x = np.linspace(0, 1, len(deletion_scores))
    # NumPy 2.0+ uses np.trapezoid, older versions use np.trapz
    auc = np.trapezoid(deletion_scores, x) if hasattr(np, 'trapezoid') else np.trapz(deletion_scores, x)
    return deletion_scores, auc

def insertion_test(image_pil, segments, importance_scores, predicted_class, model, device, num_steps=20):
    image_np = np.array(image_pil.resize((224, 224))).astype('float32')
    sorted_segments = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    segment_ids_ordered = [s[0] for s in sorted_segments]

    total_segments = len(segment_ids_ordered)
    segments_per_step = max(1, total_segments // num_steps)
    
    # Use constant gray value (mean pixel value) as insertion baseline
    # Calculate mean pixel value from training set (using ImageNet mean as approximation)
    mean_value = np.array([123.675, 116.28, 103.53])  # RGB mean values * 255 to match 0-255 scale
    # Convert to BGR since we're working with RGB images
    mean_value_bgr = mean_value[::-1]  # Reverse to get BGR order
    gray_image = np.full_like(image_np, mean_value_bgr)
    modified_image = gray_image.copy()
    transform = get_transform()

    insertion_scores = []
    pil_img = Image.fromarray(modified_image.astype('uint8'))
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        insertion_scores.append(probs[0, predicted_class].item())

    for step in range(num_steps):
        start_idx = step * segments_per_step
        end_idx = min(start_idx + segments_per_step, total_segments)
        for idx in range(start_idx, end_idx):
            seg_id = segment_ids_ordered[idx]
            mask = segments == seg_id
            modified_image[mask] = image_np[mask]

        pil_img = Image.fromarray(modified_image.astype('uint8'))
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            insertion_scores.append(probs[0, predicted_class].item())
        if end_idx >= total_segments: break

    x = np.linspace(0, 1, len(insertion_scores))
    # NumPy 2.0+ uses np.trapezoid, older versions use np.trapz
    auc = np.trapezoid(insertion_scores, x) if hasattr(np, 'trapezoid') else np.trapz(insertion_scores, x)
    return insertion_scores, auc

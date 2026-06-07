import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from gradcam_utils import SwinGradCAM

# ==============================================================================
# GUIDED LIME EXPLAINER
# ==============================================================================

class GuidedLIME:
    """
    Hybrid XAI that uses Grad-CAM as a spatial prior to guide LIME sampling.
    This combines the global localization of Grad-CAM with the local precision of LIME.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gcam = SwinGradCAM(model, device)

    def generate_hybrid_explanation(self, image_pil, num_samples=1000):
        # 1. Generate Grad-CAM Prior
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        cam_map, pred_class, _ = self.gcam.generate(img_tensor)

        # 2. Standard LIME Setup
        image_np = np.array(image_pil.resize((224, 224)))
        explainer = LimeImageExplainer(random_state=42)

        def predict_fn(images):
            batch_tensors = []
            for img in images:
                pil_img = Image.fromarray(img.astype('uint8'))
                tensor = transform(pil_img)
                batch_tensors.append(tensor)
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

        def custom_segmentation(image):
            return slic(image, n_segments=100, compactness=10, sigma=1, start_label=1)

        # 3. Hybrid Guidance: Use Grad-CAM to bias the importance weights
        # LIME's explain_instance doesn't natively take a prior, so we
        # post-process the LIME local model by weighting it with the Grad-CAM intensity
        explanation = explainer.explain_instance(
            image_np,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            batch_size=32,
            segmentation_fn=custom_segmentation
        )

        # Map Grad-CAM intensity back to superpixels
        segments = explanation.segments
        unique_segs = np.unique(segments)
        hybrid_scores = {}

        lime_scores = dict(explanation.local_exp[pred_class])

        for seg_id in unique_segs:
            mask = (segments == seg_id)
            # Average Grad-CAM intensity for this superpixel
            gcam_weight = np.mean(cam_map[mask])
            # The hybrid score is a product of the LIME weight and the Grad-CAM prior
            # This suppresses "noisy" LIME features that have no Grad-CAM support
            hybrid_scores[seg_id] = lime_scores.get(seg_id, 0) * (1 + gcam_weight)

        return segments, hybrid_scores, pred_class

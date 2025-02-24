"""
Imports models with Hugging Face API.
Author: Arogya Kharel
"""

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from transformers.image_utils import load_image
import torch

CACHE_DIR = "/workspaces/ASE-Model-Retrieval/models/data/model-cache"

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"
image = load_image(image_url)

# Use Auto classes to load a timm model
checkpoint = "timm/mobilenetv4_conv_medium.e500_r256_in1k"
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint, cache_dir=CACHE_DIR
)
model = AutoModelForImageClassification.from_pretrained(checkpoint, cache_dir=CACHE_DIR).eval()

inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

print(f"Predicted class ID: {predicted_class_idx}, Label: {model.config.id2label[predicted_class_idx]}")
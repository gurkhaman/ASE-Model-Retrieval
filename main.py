"""
Receives a meta-learning configuration and runs it.
Author: Arogya Kharel
"""

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from transformers.image_utils import load_image

CACHE_DIR = "/workspaces/ASE-Model-Retrieval/models/data/model-cache"

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"
image = load_image(image_url)

# Use Auto classes to load a timm model
checkpoint = "timm/mobilenetv4_conv_medium.e500_r256_in1k"
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint, cache_dir=CACHE_DIR
)
model = AutoModelForImageClassification.from_pretrained(checkpoint, cache_dir=CACHE_DIR).eval()

# Check the types
print(type(image_processor))  # TimmWrapperImageProcessor
print(type(model))  # TimmWrapperForImageClassification

outputs = model(image)
print(outputs)

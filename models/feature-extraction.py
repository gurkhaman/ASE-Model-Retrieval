"""
Task Embedding Extraction using CLIP and Hugging Face datasets.

Author: Arogya Kharel
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from sklearn.cluster import KMeans
import pickle
from PIL import Image

# Paths
HF_DATASET_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/hf_datasets"
LABEL_MAP_PATH = "/workspaces/ASE-Model-Retrieval/data/imagenet/label_mapping.json"
OUTPUT_DIR = "/workspaces/ASE-Model-Retrieval/task_embeddings"
BATCH_SIZE = 128
N_CLUSTERS = 1  # KMeans for meta-features

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load label mapping
with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAP = json.load(f)

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_text_features(labels):
    """
    Converts class labels into CLIP text embeddings.
    """
    text_inputs = tokenizer([f"a photo of a {label}" for label in labels], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    return text_features.cpu().numpy()

def get_image_features(image_paths):
    """
    Extracts image embeddings using CLIP.
    """
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    return image_features.cpu().numpy()

def compute_task_meta_features(features):
    """
    Computes meta-features using KMeans clustering.
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(features)
    return kmeans.cluster_centers_[0]

def process_dataset(dataset_name, dataset):
    """
    Processes an HF dataset to extract task embeddings.
    """
    print(f"Processing dataset: {dataset_name}")

    # Extract unique labels for text embeddings
    unique_labels = list(set(dataset["label"]))
    text_features = get_text_features(unique_labels)

    # Store image paths and corresponding labels
    image_paths = dataset["image"]
    labels = dataset["label"]

    # Process in batches
    image_features_list = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Extracting features for {dataset_name}"):
        batch_images = image_paths[i:i + BATCH_SIZE]
        batch_features = get_image_features(batch_images)
        image_features_list.append(batch_features)

    # Concatenate all image features
    image_features = np.vstack(image_features_list)

    # Compute meta-features
    task_meta_features = compute_task_meta_features(image_features)

    # Merge image meta-features with text features
    final_features = np.concatenate((task_meta_features, text_features.mean(axis=0)), axis=0)

    # Save features
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_task_embedding.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(final_features, f)
    
    print(f"Saved task embedding for {dataset_name}: {output_path}")

def main():
    """
    Loads datasets and processes them for task embedding extraction.
    """
    print("Loading Hugging Face datasets...")
    hf_datasets = load_from_disk(HF_DATASET_DIR)
    print(f"Loaded {len(hf_datasets)} datasets.")

    for dataset_name, dataset in hf_datasets.items():
        process_dataset(dataset_name, dataset)

if __name__ == "__main__":
    main()

"""
Task Embedding Extraction using CLIP and Hugging Face datasets.

Author: Arogya Kharel
"""

import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np


HF_DATASET_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/hf_datasets"
OUTPUT_DIR = "/workspaces/ASE-Model-Retrieval/meta-embedding/.cache/task_embeddings"
BATCH_SIZE = 128
N_CLUSTERS = 1  # KMeans for meta-features
N_SPLITS = 5  # Split features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
PROCESSOR = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_text_features(labels):
    text_inputs = TOKENIZER(
        [f"a photo of a {label}" for label in labels], padding=True, return_tensors="pt"
    ).to(DEVICE, non_blocking=True)
    with torch.no_grad():
        text_features = MODEL.get_text_features(**text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_image_features(image_paths):
    images = [Image.open(img_path).convert("RGB").copy() for img_path in image_paths]
    inputs = PROCESSOR(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        image_features = MODEL.get_image_features(**inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def compute_task_meta_features(features):
    if features.is_cuda:
        features = features.cpu().numpy()

    feature_splits = np.array_split(features, N_SPLITS)

    cluster_features = np.concatenate(
        [
            KMeans(n_clusters=N_CLUSTERS, random_state=42)
            .fit(split)
            .cluster_centers_[0]
            for split in feature_splits
        ]
    )

    return torch.tensor(cluster_features, device=DEVICE)


def process_dataset(dataset_name, dataset):
    print(f"Processing dataset: {dataset_name}")

    unique_labels = sorted(set(dataset["label"]))
    print(f"Extracting text features for {dataset_name}")
    text_features = get_text_features(unique_labels)

    image_paths = dataset["image"]
    # labels = dataset["label"]

    image_features = torch.cat(
        [
            get_image_features(image_paths[i : i + BATCH_SIZE])
            for i in tqdm(
                range(0, len(image_paths), BATCH_SIZE),
                desc=f"Extracting features for {dataset_name}",
            )
        ],
        dim=0,
    )
    task_meta_features = compute_task_meta_features(image_features)

    # Additional Feature Extraction from Image Text Descriptions
    itext_features = get_text_features(dataset["label"])  # Encode per-image labels
    itext_meta_features = compute_task_meta_features(itext_features)

    f_meta_features = torch.lerp(task_meta_features, itext_meta_features, 0.5)

    final_features = torch.cat((f_meta_features, text_features[0]), dim=0)

    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_task_embedding.pt")
    torch.save(final_features, output_path)

    print(f"Saved task embedding for {dataset_name}: {output_path}")


def main():
    hf_datasets = load_from_disk(HF_DATASET_DIR)
    print(f"Loaded {len(hf_datasets)} datasets.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset_name, dataset in tqdm(
        hf_datasets.items(), desc="Processing datasets", unit="dataset"
    ):
        process_dataset(dataset_name, dataset)


if __name__ == "__main__":
    main()

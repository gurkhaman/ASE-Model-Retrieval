import os
import json
from datasets import Dataset, DatasetDict

SUBCLASS_OUTPUT_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/subclass_datasets"
)
SUPERCLASS_OUTPUT_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/superclass_datasets"
)


def load_as_hf_dataset(image_dir):
    label_file = os.path.join(image_dir, "labels.json")
    if not os.path.exists(label_file):
        return None

    with open(label_file, "r") as f:
        labels_dict = json.load(f)

    data = []
    for img_filename, label in labels_dict.items():
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            data.append({"image": img_path, "label": label})

    return Dataset.from_list(data)


def convert_to_hf_datasets(output_path):
    all_datasets = {}

    for dataset_name in os.listdir(SUBCLASS_OUTPUT_DIR):
        dataset_path = os.path.join(SUBCLASS_OUTPUT_DIR, dataset_name)
        if os.path.isdir(dataset_path):
            dataset = load_as_hf_dataset(dataset_path)
            if dataset:
                all_datasets[f"subclass_{dataset_name}"] = dataset

    for superclass in os.listdir(SUPERCLASS_OUTPUT_DIR):
        superclass_path = os.path.join(SUPERCLASS_OUTPUT_DIR, superclass)
        if os.path.isdir(superclass_path):
            dataset = load_as_hf_dataset(superclass_path)
            if dataset:
                all_datasets[f"superclass_{superclass}"] = dataset

    hf_datasets = DatasetDict(all_datasets)
    hf_datasets.save_to_disk(output_path)

    print(f"Hugging Face datasets saved to: {output_path}")

output_hf_dataset_path = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/hf_datasets"
convert_to_hf_datasets(output_hf_dataset_path)
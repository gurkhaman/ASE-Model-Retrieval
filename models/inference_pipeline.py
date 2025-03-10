"""
Inference of all models from the list with all the datasets.

Author: Arogya Kharel
"""

import ray
import pandas as pd
from datasets import load_from_disk
from transformers import pipeline
import os


ray.init(ignore_reinit_error=True)
NUM_GPUS = int(ray.available_resources().get("GPU", 0))
if NUM_GPUS == 0:
    raise RuntimeError("No GPUs detected by Ray.")
else:
    print(f"Ray detected {NUM_GPUS} GPUs.")


# Loads dataset from disk and returns a dict
def load_hf_datasets():
    hf_dataset_dict = load_from_disk(
        "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/hf_datasets"
    )
    return hf_dataset_dict


# load models from csv
def load_models():
    models_csv = pd.read_csv("/workspaces/ASE-Model-Retrieval/models/model-list.csv")
    return models_csv


@ray.remote(num_gpus=1 / 2)
def pipeline(models, datasets):
    log_dir = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/inference_logs"
    os.makedirs(log_dir, exist_ok=True)

    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        print("[ERROR] No GPU assigned.")
        return

    gpu_id = gpu_ids[0]

    for model in models:
        for dataset in datasets:
            log_file


def inference(model, dataset):
    pass


# Main
if __name__ == "__main__":
    # hf_dataset_dict = load_hf_datasets()
    models_df = load_models()
    print(models_df)
    # print(hf_dataset_dict["image"])
    # for dataset_name, dataset in hf_dataset_dict.items():
    #     print(len(dataset["image"]))

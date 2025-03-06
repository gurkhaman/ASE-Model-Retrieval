"""
Takes in a model name and dataset name and collects evaluation metrics.

Author: Arogya Kharel
"""

from transformers import pipeline

INFERENCE_BATCH_SIZE = 128
MODEL_LIST = "/workspaces/ASE-Model-Retrieval/data/model-list.csv"
SUPERCLASS_DATASET_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/superclass_datasets"
)
SUBCLASS_DATASET_DIR = (
    "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/subclass_datasets"
)


def batch_inference(model_id, dataset, batch_size, gpu_id):
    print("Beginning batch inference for model: ", model_id)

    classifier = pipeline(model=model_id, device=gpu_id)
    return classifier

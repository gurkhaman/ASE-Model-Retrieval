"""
Inference of all models from the list with all the datasets.

Author: Arogya Kharel
"""

from datasets import load_from_disk
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import json
import warnings


HF_DATASET_DIR = "/workspaces/ASE-Model-Retrieval/data/imagenet/.cache/hf_datasets"
MODELS_CSV_PATH = "/workspaces/ASE-Model-Retrieval/models/model-list.csv"
LABEL_MAP_PATH = "/workspaces/ASE-Model-Retrieval/data/imagenet/label_mapping.json"
BATCH_SIZE = 128
TASK_PER_GPU = 2

MODEL_PIPELINES = {}
with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAP = json.load(f)


def get_pipeline(model_id, gpu_id):
    if model_id not in MODEL_PIPELINES:
        MODEL_PIPELINES[model_id] = pipeline(model=model_id, device=gpu_id)
    return MODEL_PIPELINES[model_id]


def batch_classification(images, model_id, gpu_id, batch_size=128):
    print(f"Batch classification for {model_id}.")
    classifier = get_pipeline(model_id, gpu_id)

    predictions = []

    for img_path, pred in zip(
        images, tqdm(classifier(images, batch_size=batch_size), total=len(images), colour="blue")
    ):
        corrected_pred = [
            {"label": LABEL_MAP.get(p["label"], p["label"]), "score": p["score"]}
            for p in pred
        ]
        predictions.append(
            {"image": os.path.basename(img_path), "prediction": corrected_pred}
        )

    return predictions


def evaluate_model(predictions, dataset):
    ground_truth = {os.path.basename(item["image"]): item["label"] for item in dataset}

    y_true, y_pred, y_pred_top5 = zip(
        *[
            (
                ground_truth[entry["image"]],
                entry["prediction"][0]["label"],
                [pred["label"] for pred in entry["prediction"]],
            )
            for entry in predictions
            if entry["image"] in ground_truth
        ]
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    top1_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    top5_accuracy = np.mean([true in preds for true, preds in zip(y_true, y_pred_top5)])

    return {
        "Top-1 Accuracy": top1_accuracy,
        "Top-5 Accuracy": top5_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Classification Report": classification_report(
            y_true, y_pred, output_dict=True
        ),
    }


def process_model(model_id):
    results = []
    for dataset_name, dataset in tqdm(
        hf_dataset_dict.items(),
        desc=f"Processing {model_id}",
        colour="green",
        leave=True,
    ):
        print(f"Evaluating {model_id} on {dataset_name}")
        predictions = batch_classification(dataset["image"], model_id, gpu_id=0)
        result = evaluate_model(predictions, dataset)
        results_df = pd.DataFrame(
            [{"model_id": model_id, "dataset": dataset_name, **result}]
        )

        file_path = f"evaluation_results/{dataset_name}.csv"
        if os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            results_df.to_csv(file_path, index=False)

        print(f"Results saved: {file_path}")

    return results


if __name__ == "__main__":
    hf_dataset_dict = load_from_disk(HF_DATASET_DIR)
    print(f"Loaded {len(hf_dataset_dict.items())} datasets...")
    models_df = pd.read_csv(MODELS_CSV_PATH)
    print(f"Loaded {models_df.shape[0]} pre-trained models...")

    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs("evaluation_results", exist_ok=True)

    tqdm.pandas(desc="Processing Models", colour="yellow")
    models_df["evaluation_results"] = models_df["model-id"].progress_apply(
        process_model
    )
    log_results = [
        entry for sublist in models_df["evaluation_results"] for entry in sublist
    ]
    results_df = pd.DataFrame(log_results)
    results_df.to_csv("model_evaluation_log.csv", index=False)

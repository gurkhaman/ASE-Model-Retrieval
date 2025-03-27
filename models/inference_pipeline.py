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
import yaml
import argparse
import ray


def get_pipeline(model_id, gpu_id, pipeline_cache):
    if model_id not in pipeline_cache:
        pipeline_cache[model_id] = pipeline(model=model_id, device=gpu_id)
    return pipeline_cache[model_id]


def batch_classification(
    images, model_id, gpu_id, label_map, batch_size, pipeline_cache
):
    print(f"Batch classification for {model_id}.")
    classifier = get_pipeline(model_id, gpu_id, pipeline_cache)

    predictions = []
    for img_path, pred in zip(
        images,
        tqdm(
            classifier(images, batch_size=batch_size), total=len(images), colour="blue"
        ),
    ):
        corrected_pred = [
            {"label": label_map.get(p["label"], p["label"]), "score": p["score"]}
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
    }


def aggregate_results(repetitions, dataset_names, output_dir):
    aggregated_rows = []

    for dataset in dataset_names:
        all_dfs = [
            pd.read_csv(os.path.join(output_dir, f"run_{r}", f"{dataset}.csv"))
            for r in range(1, repetitions + 1)
            if os.path.exists(os.path.join(output_dir, f"run_{r}", f"{dataset}.csv"))
        ]
        if all_dfs:
            combined_df = pd.concat(all_dfs)
            grouped = (
                combined_df.groupby(["model_id", "dataset"])
                .agg(
                    {
                        "Top-1 Accuracy": ["mean", "std"],
                        "Top-5 Accuracy": ["mean", "std"],
                        "Precision": ["mean", "std"],
                        "Recall": ["mean", "std"],
                        "F1-score": ["mean", "std"],
                    }
                )
                .reset_index()
            )

            grouped.columns = [
                "_".join(col).strip("_") for col in grouped.columns.values
            ]
            aggregated_rows.append(grouped)

    if aggregated_rows:
        final_df = pd.concat(aggregated_rows)
        final_df.to_csv(os.path.join(output_dir, "aggregated_metrics.csv"), index=False)
        print(
            f"Aggregated metrics saved to: {os.path.join(output_dir, 'aggregated_metrics.csv')}"
        )


def process_model(
    model_id, dataset_dict, label_map, batch_size, pipeline_cache, run_dir
):
    results = []
    for dataset_name, dataset in tqdm(
        dataset_dict.items(), desc=f"Processing {model_id}", colour="green", leave=True
    ):
        print(f"Evaluating {model_id} on {dataset_name}")
        predictions = batch_classification(
            dataset["image"],
            model_id,
            gpu_id=0,
            label_map=label_map,
            batch_size=batch_size,
            pipeline_cache=pipeline_cache,
        )
        result = evaluate_model(predictions, dataset)
        results_df = pd.DataFrame(
            [{"model_id": model_id, "dataset": dataset_name, **result}]
        )

        file_path = os.path.join(run_dir, f"{dataset_name}.csv")
        if os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            results_df.to_csv(file_path, index=False)

        print(f"Results saved: {file_path}")

    return results


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluation with config")
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    return parser.parse_args()


@ray.remote(num_gpus=1 / 3)
def process_model_ray(model_id, dataset_dict, label_map, batch_size, run_dir):
    # Make sure run_dir exists inside the worker
    os.makedirs(run_dir, exist_ok=True)
    pipeline_cache = {}
    return process_model(
        model_id, dataset_dict, label_map, batch_size, pipeline_cache, run_dir
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    paths = config["paths"]
    params = config["params"]
    evaluation_results_dir = paths["evaluation_results_dir"]

    batch_size = params.get("batch_size", 128)
    repetitions = params.get("repetitions", 1)
    tasks_per_gpu = params.get("task_per_gpu", 1)

    # Load label map
    with open(paths["label_map"], "r") as f:
        label_map = json.load(f)

    # Load dataset
    dataset_dict = load_from_disk(paths["dataset_dir"])
    dataset_names = list(dataset_dict.keys())
    print(f"Loaded {len(dataset_dict.items())} datasets...")

    # Load models CSV
    models_df = pd.read_csv(paths["models_csv"])
    print(f"Loaded {models_df.shape[0]} pre-trained models...")

    # Setup
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(evaluation_results_dir, exist_ok=True)

    ray.init()
    NUM_GPUS = ray.available_resources().get("GPU", 1)
    MAX_CONCURRENT_TASKS = int(NUM_GPUS * tasks_per_gpu)

    tqdm.pandas(desc="Processing Models", colour="yellow")

    for rep in range(repetitions):
        print(f"\n===== Starting run {rep + 1} of {repetitions} =====")

        run_dir = os.path.join(evaluation_results_dir, f"run_{rep + 1}")

        tasks = [
            process_model_ray.remote(
                model_id, dataset_dict, label_map, batch_size, run_dir
            )
            for model_id in models_df["model-id"]
        ]

        remaining_tasks = tasks
        results = []
        while remaining_tasks:
            num_ready = min(MAX_CONCURRENT_TASKS, len(remaining_tasks))
            done_tasks, remaining_tasks = ray.wait(
                remaining_tasks, num_returns=num_ready
            )
            results.extend(ray.get(done_tasks))

    aggregate_results(repetitions=repetitions, dataset_names=dataset_names)


if __name__ == "__main__":
    main()

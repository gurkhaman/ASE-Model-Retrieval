import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


def load_meta_features(path, device):
    all_features = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    print(f"Found {len(all_features)} meta-feature files.")

    meta_features = torch.stack(
        [
            torch.load(os.path.join(path, feature), map_location=device).flatten()
            for feature in all_features
        ],
        dim=0,
    )
    return meta_features


def load_evaluation_data(path):
    all_results = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    print(f"Found {len(all_results)} evaluation data files.")

    all_results = [
        pd.read_csv(os.path.join(path, result), delimiter=",") for result in all_results
    ]

    return all_results


def extract_evaluation_labels(evaluation_data, column):
    evaluation_labels = torch.tensor(
        [torch.tensor(df[column].values).argmax().item() for df in evaluation_data]
    )

    return evaluation_labels


def extract_real_accuracy_matrix(evaluation_data):
    real_accuracy_matrix = torch.stack(
        [torch.tensor(df["Top-1 Accuracy"].values).float() for df in evaluation_data],
        dim=1,
    ).T

    return real_accuracy_matrix


def load_data(meta_features_dir, eval_results_dir, device):
    meta_features = load_meta_features(meta_features_dir, device)
    evaluation_data = load_evaluation_data(eval_results_dir)

    X = meta_features.cpu().numpy()
    Y = extract_evaluation_labels(evaluation_data, "Top-1 Accuracy")

    real_accuracy_matrix = extract_real_accuracy_matrix(evaluation_data)

    X_train, X_test, y_train, y_test, idx_train, idx_test = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, Y, range(len(meta_features)), test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        idx_train,
        idx_test,
        meta_features,
        evaluation_data,
        real_accuracy_matrix,
    )

from dataclasses import dataclass
import wandb
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class LearnerData:
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    idx_train: any
    idx_test: any
    meta_features: any
    evaluation_data: any
    real_accuracy_matrix: any


def load_meta_features(meta_features_dir):
    all_features = sorted(
        [f for f in os.listdir(meta_features_dir) if f.endswith(".pt")]
    )
    wandb.log({"num_meta_features": len(all_features)})

    meta_features = torch.stack(
        [
            torch.load(
                os.path.join(meta_features_dir, feature)
            ).flatten()  # FIXME: add map_location=device if necessary
            for feature in all_features
        ],
        dim=0,
    )

    return meta_features


def filter_df(df, exclude, include):
    if "model_id" not in df.columns:
        return pd.DataFrame()

    if include:
        df = df[df["model_id"].isin(include)]

    if exclude:
        df = df[~df["model_id"].isin(exclude)]

    return df


def extract_evaluation_labels(evaluation_data, target):
    """
    Extract a column from the evaluation files to be selected as the label.
    """
    evaluation_labels = torch.tensor(
        [torch.tensor(df[target].values).argmax().item() for df in evaluation_data]
    )

    return evaluation_labels


def extract_real_accuracy_matrix(evaluation_data, target):
    real_accuracy_matrix = torch.stack(
        [torch.tensor(df[target].values).float() for df in evaluation_data],
        dim=1,
    ).T

    return real_accuracy_matrix


def load_evaluation_data(evaluation_results_dir, exclude, include):
    exclude = set(exclude or [])
    print("Excluded models:", exclude)
    wandb.log({"num_excluded": len(exclude)})

    include = set(include or [])
    print("Include_only models:", include)
    wandb.log({"num_include_only": len(include)})

    all_eval_files = sorted(
        [f for f in os.listdir(evaluation_results_dir) if f.endswith(".csv")]
    )
    wandb.log({"num_eval_data": len(all_eval_files)})

    all_eval_data = []

    for filename in all_eval_files:
        file_path = os.path.join(evaluation_results_dir, filename)
        df = pd.read_csv(file_path)

        df = filter_df(df, exclude, include)

        if not df.empty:
            all_eval_data.append(df)

    return all_eval_data


def load_data(
    meta_features_dir, evaluation_results_dir, exclude, include
) -> LearnerData:
    meta_features = load_meta_features(meta_features_dir)
    evaluation_data = load_evaluation_data(evaluation_results_dir, exclude, include)
    evaluation_labels = extract_evaluation_labels(
        evaluation_data, target="Top-1 Accuracy"
    )
    real_accuracy_matrix = extract_real_accuracy_matrix(
        evaluation_data, target="Top-1 Accuracy"
    )

    X = meta_features.cpu().numpy()
    Y = evaluation_labels.cpu().numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, Y, range(len(meta_features)), test_size=0.2, random_state=42
    )

    return LearnerData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        idx_train=idx_train,
        idx_test=idx_test,
        meta_features=meta_features,
        evaluation_data=evaluation_data,
        real_accuracy_matrix=real_accuracy_matrix,
    )

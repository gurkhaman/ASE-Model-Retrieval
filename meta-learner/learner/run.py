import argparse
import torch
from learner.data.load_data import load_data
from learner.train.train_meta import train_and_evaluate
from learner.utils.config import load_config
from learner.train.evaluate_rank import evaluate_ranking
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    model_cfg = config["model"]
    wandb_cfg = config.get("wandb", {})

    wandb.init(
        project=wandb_cfg.get("project", "meta-learner"),
        name=wandb_cfg.get("run_name", f"{model_cfg['name']}_run"),
        config={"model": model_cfg["name"], **model_cfg.get("params", {})},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        idx_train,
        idx_test,
        _,
        _,
        real_accuracy_matrix,
    ) = load_data(paths["meta_features_dir"], paths["evaluation_results_dir"], device)

    results = train_and_evaluate(
        model_name=model_cfg["name"],
        model_params=model_cfg.get("params", {}),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        use_wandb=wandb_cfg.get("enabled", False),
    )
    print("Accuracy:", results["accuracy"])
    print("Classification Report:")
    for label, scores in results["report"].items():
        print(f"{label}: {scores}")

    if results.get("y_proba") is not None:
        summary = evaluate_ranking(
            y_proba=results["y_proba"],
            idx_test=idx_test,
            real_accuracy_matrix=real_accuracy_matrix,
            num_classes=model_cfg["params"]["num_classes"],
        )

        print("Ranking Evaluation Summary:")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}")

        if wandb_cfg.get("enabled", False):
            import numpy as np

            entropy = (
                (-results["y_proba"] * np.log(results["y_proba"] + 1e-9))
                .sum(axis=1)
                .mean()
            )
            wandb.log({"accuracy": results["accuracy"]})
            wandb.log({"mean_entropy": entropy})
            wandb.log(
                {
                    f"ranking/{k}": v
                    for k, v in summary.items()
                    if isinstance(v, (int, float))
                }
            )


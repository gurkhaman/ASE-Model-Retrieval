from evaluation.evaluator import Evaluator, Model


def evaluate_ranking(y_proba, idx_test, real_accuracy_matrix, num_classes):
    evaluator = Evaluator(k_ndcg=5, k_map=3)

    for i, task_index in enumerate(idx_test):
        model_list = [
            Model(
                model_id=j,
                pred_perf=y_proba[i][j],
                real_perf=real_accuracy_matrix[task_index][j].item(),
            )
            for j in range(num_classes)
        ]
        evaluator.evaluate_task(model_list)

    return evaluator.summarize()

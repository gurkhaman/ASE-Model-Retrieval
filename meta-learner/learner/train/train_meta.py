from learner.core.factory import load_model
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(model_name, model_params, X_train, y_train, X_test, y_test):
    model = load_model(model_name, **model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Optional proba support for top-k eval/logging
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = XGBClassifier(**xgb_params)
        self.classes_ = None
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder_.fit(y)
        self.classes_ = self.label_encoder_.classes_
        y_encoded = self.label_encoder_.transform(y)

        self.model.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded_pred = self.model.predict(X)
        return self.label_encoder_.inverse_transform(y_encoded_pred)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        trained_classes = self.model.classes_

        if len(trained_classes) == len(self.classes_) and np.all(trained_classes == np.arange(len(self.classes_))):
            return proba  # All classes included, already aligned

        # Pad to full class list
        full_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls_index in enumerate(trained_classes):
            full_proba[:, cls_index] = proba[:, i]

        return full_proba

from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomBaggingSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)
        self.estimators_ = []

        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_sub, y_sub = X[indices], y[indices]

            clf = SVC(probability=True)
            clf.fit(X_sub, y_sub)
            self.estimators_.append(clf)

        return self

    def predict_proba(self, X):
        all_proba = []

        for clf in self.estimators_:
            proba = clf.predict_proba(X)

            full_proba = np.zeros((X.shape[0], self.num_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_index = np.where(self.classes_ == cls)[0][0]
                full_proba[:, cls_index] = proba[:, i]

            all_proba.append(full_proba)

        return np.mean(all_proba, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

from learner.core.base_model import MetaModel
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import stats


class CustomBaggingSVC(MetaModel):
    def __init__(self, n_estimators=10, num_classes=93):
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.estimators = []
        self.scaler = None

    def fit(self, X_train, y_train):
        self.scaler = StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        self.estimators = []
        for seed in range(self.n_estimators):
            np.random.seed(seed)
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_sub = X_scaled[indices]
            y_sub = y_train[indices]

            clf = SVC(probability=True)
            clf.fit(X_sub, y_sub)
            self.estimators.append(clf)

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        all_probs = []

        for clf in self.estimators:
            proba = clf.predict_proba(X_scaled)

            probs_full = np.zeros((X_test.shape[0], self.num_classes))
            for i, cls in enumerate(clf.classes_):
                probs_full[:, cls] = proba[:, i]

            all_probs.append(probs_full)

        return np.mean(all_probs, axis=0)

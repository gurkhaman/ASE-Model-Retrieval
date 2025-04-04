class MetaModel:
    def fit(self, X_train, y_train):
        raise NotImplementedError("fit() must be implemented by subclasses")

    def predict(self, X_test):
        raise NotImplementedError("predict() must be implemented by subclasses")

    def predict_proba(self, X_test):
        """
        Optional: Used for models that support confidence/ranking outputs.
        If not implemented, fall back to predict().
        """
        return None

    def save(self, path):
        """
        Optional: Save the model to disk.
        """
        pass

    def load(self, path):
        """
        Optional: Load the model from disk.
        """
        pass

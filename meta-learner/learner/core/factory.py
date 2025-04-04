from learner.models.custom_bagging_svc import CustomBaggingSVC


def load_model(model_name, **kwargs):
    if model_name == "custom_bagging_svc":
        return CustomBaggingSVC(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

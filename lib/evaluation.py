from sklearn.metrics import f1_score


def get_f1_score(true, pred, average="micro"):
    return f1_score(true, pred, average=average)


def evaluate(true, pred, eval_name):
    if eval_name == "F1-score":
        return get_f1_score(true, pred, average="micro")
    raise ValueError(f"{eval_name} is not implemented.")

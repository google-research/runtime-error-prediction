from sklearn.metrics import f1_score, confusion_matrix

def get_f1_score(true, pred, average="micro"):
  return f1_score(true, pred, average=average)

def get_confusion_matrix_score(true, pred):
  return confusion_matrix(true, pred)

def evaluate(true, pred, eval_name):
  if eval_name == "F1-score":
    return get_f1_score(true, pred, average="micro")
  elif eval_name == "Confusion matrix":
    return get_confusion_matrix_score(true, pred)
  raise ValueError(f"{eval_name} is not implemented.")

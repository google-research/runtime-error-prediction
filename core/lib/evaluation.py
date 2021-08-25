from sklearn import metrics


def evaluate(true, pred, eval_name):
  if eval_name == 'F1-score':
    # TODO(dbieber): Support macro f1.
    return metrics.f1_score(true, pred, average='micro')
  elif eval_name == 'Confusion matrix':
    return metrics.confusion_matrix(true, pred)
  raise ValueError(f'{eval_name} is not implemented.')

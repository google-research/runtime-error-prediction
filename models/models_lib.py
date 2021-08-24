from models import rnn_model, transformers, ipa_gnns


class ModelFactory:
  def __init__(self):
    super(ModelFactory, self).__init__()
    self._models = {"StackedLSTMModel": rnn_model.StackedLSTMModel,
      "IPAGNN":ipa_gnns.IPAGNN,
      "Transformer": transformers.Transformer}

  def register(self, model_name, model_cls):
    self._models[model_name] = model_cls

  def __call__(self, model_name):
    if model_name not in self._models:
      raise ValueError(f"Model {model_name} is not defined.")
    return self._models[model_name]

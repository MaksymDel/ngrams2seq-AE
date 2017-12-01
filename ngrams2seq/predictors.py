from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('simple_ngrams2seq')
class Ngrams2SeqPredictor(Predictor):
    """
    Wrapper for the simple_ngram2seq model.
    """
    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"text": "..."}``.
        """
        src = json["sentence"]
        return self._dataset_reader.text_to_instance(src)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self._model.forward_on_instances(instances, cuda_device)
        return sanitize(outputs["predicted_tokens"])

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        print('AM', len(sanitize(outputs["attention_matrix"])))
        return sanitize(outputs["predicted_tokens"])

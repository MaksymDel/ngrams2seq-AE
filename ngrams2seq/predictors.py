from typing import List
from overrides import overrides
import numpy as np
from nltk import everygrams

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('ngrams2seq')
class Ngrams2SeqPredictor(Predictor):
    """
    Wrapper for the ngram2seq model.
    """
    def save_attention_matrix(self, att_matrix: np.ndarray, src_tokens, tgt_tokens) -> None:
        """
        Write attention matrix to the disc under att_matrix.txt name in Nematus like format
        for future visualization.

        This function is dummy and should be rewritten using config ngram_order parameter,
        configurable filename, and probably should open file one time before prediction
        in other external place.
        """
        print('\nSaving attention matrix of shape', att_matrix.shape, 'as att_matrix.txt...')

        filename = 'att_matrix.txt'

        # convert src sentece to ngrams;
        MAX_NGRAM_DEGREE = 2
        ngrams_iterator = everygrams(src_tokens.split(), max_len=MAX_NGRAM_DEGREE)
        
        src_tokens = " ".join(["_".join(ngram) for ngram in ngrams_iterator])
        tgt_tokens = " ".join(tgt_tokens)

        num_tgt = str(len(tgt_tokens.split()) + 1)
        num_src = str(len(src_tokens.split()) + 1)
        
        # add a column of zeros to att matrix as required by Nematus visualization tool
        att_matrix = np.c_[att_matrix, np.zeros(att_matrix.shape[0])]
        # truncate redundent raws that correspond to EOS tokens
        att_matrix = att_matrix[:int(num_tgt), :]

        with open(filename, mode='a') as f_handle:
            header = '0 ||| ' + tgt_tokens + ' ||| 0 ||| ' + src_tokens + ' ||| ' + num_src + ' ' + num_tgt + '\n'
            f_handle.write(header)

        with open(filename, mode='ab') as f_handle:
            np.savetxt(f_handle, att_matrix)
        
        with open(filename, mode='a') as f_handle:
            f_handle.write('\n')
            
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

        self.save_attention_matrix(outputs["attention_matrix"], inputs['sentence'], outputs["predicted_tokens"])

        return sanitize(outputs["predicted_tokens"])

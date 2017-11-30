from nltk import ngrams
from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("ngram")
class NgramTokenizer(Tokenizer):
    """
    A ``NgramTokenizer`` splits strings into ngram tokens.
    It assumes that sentece is already pre-tokenized e.g. with moses tokenizer, 
    so that we can easily split by space here.
   
    Parameters
    ----------
    ngram_degree : int
    'N' parameter in ngram definition. 
    It represents how many words each ngram should contain,  e.g. 2 for 2gram, 3 for 3gram, etc.
    """
    def __init__(self,
                 ngram_degree: int) -> None:
        self._ngram_degree = ngram_degree

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Splits words to ngrams using nltk
        """
        ngrams_iterator = ngrams(text.split(), self._ngram_degree)
        tokens = [Token(" ".join(ngram)) for ngram in ngrams_iterator]
        return tokens    

    @classmethod
    def from_params(cls, params: Params) -> 'NgramTokenizer':
        ngram_degree = params.pop('ngram_degree')
        params.assert_empty(cls.__name__)
        return cls(ngram_degree=ngram_degree)

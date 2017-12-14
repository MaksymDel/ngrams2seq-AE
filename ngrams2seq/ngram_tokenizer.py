from typing import List

from nltk import everygrams
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
    max_ngram_degree : int
    'N' parameter in ngram definition. 
    It represents how many words each ngram should contain,  e.g. 2 for 2gram, 3 for 3gram, etc.
    """
    # TODO(maxdel): add option for option left and rigth <SOS> and <EOS> symbols for each ngram.
    # It can be easily done with nltk's ngram function (it has corresponding formal arguments)
    def __init__(self,
                 max_ngram_degree: int,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._max_ngram_degree = max_ngram_degree
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Splits sentences into a set of all possible ngrams up to self._max_ngram_degree using nltk
        """
        ngrams_iterator = everygrams(text.split(), max_len=self._max_ngram_degree)
        tokens = [Token(" ".join(ngram)) for ngram in ngrams_iterator]
        for start_token in self._start_tokens:
            tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            tokens.append(Token(end_token, -1))
        return tokens    

    @classmethod
    def from_params(cls, params: Params) -> 'NgramTokenizer':
        max_ngram_degree = params.pop('max_ngram_degree')
        params.assert_empty(cls.__name__)
        return cls(max_ngram_degree=max_ngram_degree)

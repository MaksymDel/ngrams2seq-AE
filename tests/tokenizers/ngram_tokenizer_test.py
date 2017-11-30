# pylint: disable=no-self-use,invalid-name

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from ngrams2seq.tokenizers import NgramTokenizer

class TestNgramTokenizer(AllenNlpTestCase):
    def test_passes_through_correctly(self):
        tokenizer = NgramTokenizer(max_ngram_degree=2)
        sentence = "i go to school"
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = ["i", "go", "to", "school", "i go", 'go to', 'to school']
        assert tokens == expected_tokens

    def test_from_params_works_correctly(self):        
        tokenizer = NgramTokenizer.from_params(Params({'max_ngram_degree': 2}))
        assert tokenizer._max_ngram_degree == 2

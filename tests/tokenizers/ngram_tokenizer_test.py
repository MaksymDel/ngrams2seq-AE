# pylint: disable=no-self-use,invalid-name

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from ngrams2seq.tokenizers import NgramTokenizer

class TestNgramTokenizer(AllenNlpTestCase):
    def test_passes_through_correctly(self):
        tokens = None
        expected_tokens = None

        assert tokens == expected_tokens

    def test_from_params_ngram_degree_option_passes_correctly(self):
        tokens = None
        expected_tokens = None

        assert tokens == expected_tokens
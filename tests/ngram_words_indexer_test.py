# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from ngrams2seq.ngram_words_indexer import NgramWordsIndexer

class NgramWordsIndexerTest(AllenNlpTestCase):
    def test_count_vocab_items_works_properly(self):
        indexer = NgramWordsIndexer("words")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("i go to school"), counter)
        indexer.count_vocab_items(Token("every day i go"), counter)
        assert counter["words"] == {"i": 2, "go": 2, "to": 1, "school": 1, "every": 1, "day": 1}

    def test_as_array_produces_token_sequence(self):
        indexer = NgramWordsIndexer("words")
        padded_tokens = indexer.pad_token_sequence([[1, 2, 3, 4, 5], [1, 2, 3], [1]],
                                                   desired_num_tokens=4,
                                                   padding_lengths={"num_ngram_words": 10})
        assert padded_tokens == [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                 [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_token_to_indices_produces_correct_words(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("i", namespace='words')
        vocab.add_token_to_namespace("go", namespace='words')
        vocab.add_token_to_namespace("to", namespace='words')
        vocab.add_token_to_namespace("school", namespace='words')
        vocab.add_token_to_namespace("every", namespace='words')
        vocab.add_token_to_namespace("day", namespace='words')

        indexer = NgramWordsIndexer("words")
        indices = indexer.token_to_indices(Token("i go to collage month every , Yoda"), vocab)
        assert indices == [2, 3, 4, 1, 1, 6, 1, 1]
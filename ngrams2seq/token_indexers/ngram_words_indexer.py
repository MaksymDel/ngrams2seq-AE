from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

@TokenIndexer.register("words")
class NgramWordsIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents ngrams (which are tokens) as lists of word indices.
    It assumes words in ngram are already tokenized externaly (e.g. with Moses tokenizer)
    and can be just split by space.

    Parameters
    ----------
    namespace : ``str``, optional (default=``shared_words_vocab``)
        We will use this namespace in the :class:`Vocabulary` to map the words in each token
        to indices.
    word_tokenizer : `WordTokenizer`, optional (default=`WordTokenizer(word_splitter=JustSpacesWordSplitter())`)
        Defines the way we split ngram to words. Default is just to split by space.
        """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'shared_words_vocab',
                 word_tokenizer: WordTokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())) -> None:
        self._namespace = namespace
        self._word_tokenizer = word_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('NgramWordIndexer needs a tokenizer that retains text')
        for word in self._word_tokenizer.tokenize(token.text):
            # If `text_id` is set on the word token (e.g., if we're using custom vocab.), we
            # will not be using the vocab for this word.
            if getattr(word, 'text_id', None) is None:
                counter[self._namespace][word.text] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        indices = []
        if token.text is None:
            raise ConfigurationError('NgramWordIndexer needs a tokenizer that retains text')
        for word in self._word_tokenizer.tokenize(token.text):
            if getattr(word, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just
                # use this id instead.
                index = word.text_id
            else:
                index = vocabulary.get_token_index(word.text, self._namespace)
            indices.append(index)
        return indices

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_ngram_words': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        padded_tokens = pad_sequence_to_length(tokens, desired_num_tokens, default_value=lambda: [])
        desired_token_length = padding_lengths['num_ngram_words']
        longest_token: List[int] = max(tokens, key=len, default=[])
        padding_index = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" to get the speed of itertools.zip_longest.
            padded_tokens.append([padding_index] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_index)))
        if desired_token_length > len(longest_token):
            # now we remove the "dummy token" if we appended one.
            padded_tokens.pop()

        # Now we need to truncate all of them to our desired length, and return the result.
        return [list(token[:desired_token_length]) for token in padded_tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'NgramWordsIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``shared_words_vocab``)
            We will use this namespace in the :class:`Vocabulary` to map the words in each token
            to indices.
        word_tokenizer : `WordTokenizer`, optional (default=`WordTokenizer(word_splitter=JustSpacesWordSplitter())`)
            Defines the way we split ngram to words. Default is just to split by space.
        """
        namespace = params.pop('namespace', 'shared_words_vocab')
        word_tokenizer_params = params.pop('word_tokenizer', {})
        word_tokenizer = WordTokenizer.from_params(word_tokenizer_params)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, word_tokenizer=word_tokenizer)

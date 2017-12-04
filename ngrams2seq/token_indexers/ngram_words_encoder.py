import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("word_encoding")
class NgramWordsEncoder(TokenEmbedder):
    """
    A ``NgramWordsEncoder`` takes the output of a
    :class:`~allennlp.data.token_indexers.NgramWordsIndexer`, which is a tensor of shape
    (batch_size, num_ngrams, num_words), embeds the words, runs a ngram-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_ngrams, encoding_dim).  We also
    optionally apply dropout after the ngram-level encoder.

    We take the embedding and encoding modules as input, so this class is itself quite simple.
    """
    def __init__(self, embedding: Embedding, encoder: Seq2VecEncoder, dropout: float = 0.0) -> None:
        super(NgramWordsEncoder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()  # pylint: disable=protected-access

    def forward(self, ngram_words: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ

        mask = (ngram_words != 0).long()
        return self._dropout(self._encoder(self._embedding(ngram_words), mask))

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'NgramWordsEncoder':
        embedding_params: Params = params.pop("embedding")
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "ngram_words" by default.
        embedding_params.setdefault("vocab_namespace", "ngram_words")
        embedding = Embedding.from_params(vocab, embedding_params)
        encoder_params: Params = params.pop("encoder")
        encoder = Seq2VecEncoder.from_params(encoder_params)
        dropout = params.pop("dropout", 0.0)
        params.assert_empty(cls.__name__)
        return cls(embedding, encoder, dropout)

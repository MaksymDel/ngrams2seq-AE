import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.common import Params, Registrable

@Seq2SeqEncoder.register("bypass_seq2seq")
class BypassSeq2SeqEncoder(Seq2SeqEncoder, Registrable):
    """
    A ``BypassSeq2SeqEncoder`` is a ``Module`` that takes as input a sequence of vectors 
    and just returns this sequence itself without any modifications.
    Input shape: ``(batch_size, sequence_length, input_dim)``; output
    shape: ``(batch_size, sequence_length, input_dim)``.

    We add two methods to the basic ``Module`` API: :func:`get_input_dim()` and :func:`get_output_dim()`.
    You might need this if you want to construct a ``Linear`` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.

    Parameters
    ----------
    input_dim : ``int``
        The dimension of the vector for each element in the input sequence;
        ``input_tensor.size(-1)``.
    """

    def __init__(self, 
                 input_dim : int) -> None:
        super(BypassSeq2SeqEncoder, self).__init__()
        self._input_dim = input_dim

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_dim

    def get_output_dim(self) -> int:
        """
        The same as input_dim.
        Returns the dimension of each vector in the sequence output by this ``BypassSeq2SeqEncoder``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self._input_dim

    def forward(self,  # pylint: disable=arguments-differ
            inputs: torch.Tensor) -> torch.Tensor:
            """
            Just passes inputs forward as they are
            """
            return inputs

    @classmethod
    def from_params(cls, params: Params) -> 'BypassSeq2SeqEncoder':
        input_dim = params.pop('input_dim')
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim)

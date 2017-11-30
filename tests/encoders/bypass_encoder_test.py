# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase

from ngrams2seq.encoders.bypass_encoder import BypassSeq2SeqEncoder

class BypassSeq2SeqEncoderTest(AllenNlpTestCase):
    def test_bypass_passes_without_change(self):
        encoder = BypassSeq2SeqEncoder(input_dim = 2)
        input = numpy.array([[1.0],[2.0],[3.0],[4.0]], dtype='f')
        input_tensor = Variable(torch.FloatTensor(input))
        encoded = encoder(input_tensor).data.numpy()
        assert_equal(encoded, input) 
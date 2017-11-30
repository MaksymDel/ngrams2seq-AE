# ngrams2seq AE
This experment proposes autoencoder that generates sentence from its bag of ngrams representation      

To implement the idea we use AllenNLP library. 

The problem can be representet in terms of allennlp abstractions following way:
1) [Token]: ngram
2) [TokenIndexer]: indexes words that compose ngram for each ngram
3) [Filed]: source TextFiled based on ngram tokens; target TextField based on word tokens
4) [TokenEmbedder]: embeds each ngram words separatly, and then computes single vector representation for each ngram with RNN
5) [Encoder]: just passes bag of ngrams forward (bypass encoder)
6) [AttentionalDecoder]: takes encoder outputs (bag of ngrams), and tryies to reconstruct orgiginal sentence based on it
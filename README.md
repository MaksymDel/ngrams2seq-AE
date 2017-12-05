# ngrams2seq AE

### Dependencies
1) Fetch allennlp version specified in requirements.txt
2) [optional] Get also https://github.com/M4t1ss/SoftAlignments if you want to visualize attention weights

### Visualizing attentions
* Fetch https://github.com/M4t1ss/SoftAlignments.
* Attention matrix, which is an input to the tool, saves automaticaly under the name `att_matrix.txt`.
* You can run a tool (e.g.) as a web server using following command (from SoftAlignments repository root folder): 
`python process_alignments.py -i path_to_ngrams2seq/att_matrix.txt -o web -f Nematus`

### TODO
This experment proposes autoencoder that generates sentence from its bag of ngrams representation      

To implement the idea we use AllenNLP library. 

The problem can be representet in terms of allennlp abstractions following way:
1) [Token]: ngram -- DONE
2) [TokenIndexer]: indexes words that compose ngram for each ngram -- DONE
3) [Filed]: source TextFiled based on ngram tokens; target TextField based on word tokens -- DONE
4) [TokenEmbedder]: for each ngram embeds each word separatly, and then runs RNN over word embeddings to get ngram embedding -- DONE
5) [Encoder]: just passes bag of ngrams forward (bypass encoder) -- DONE
6) [AttentionalDecoder]: takes encoder outputs (bag of ngrams), and tryies to reconstruct orgiginal sentence based on it -- DONE

Allennlp Seq2SeqDatasetReader is probably can be used out of the box here, since it accepts source and target Tokenizers and TokenIndexers as constructor parameters.

We just need to implement our custom NgramTokenzer [DONE] and NgramIndexer [DONE] and pass them as a source side parameters to seq2seq dataset. Target side (sentence) parameters should be WordTokenizer, and SingleIdIndexer. 

NgramIndexer should be something like MultipleIdIndexer or so and it should probably use Word tokenizer. It definitly should use shared with target side words vocabulary.  
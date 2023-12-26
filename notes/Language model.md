
# Architecture (transformers)

## Tokenizer
Should not be there, character should be a unit of token.

## [[Encoder]]
given a token, that is character convert it to a vector, from a map

## [[Transformer]]
token can look other token and change their value by taking data from other tokens

## [[Embedding]]
Tokens can know where they are in a sentence.
Ideally every token should be centre and should know relative position of other token.
### Evaluation
* Text Search:  https://github.com/beir-cellar/beir/wiki
* code search: https://github.com/github/CodeSearchNet
* sentence similarity: https://github.com/facebookresearch/SentEval (STS 2012–2016)
* text classification: https://github.com/facebookresearch/SentEval (MR, CR, SUBJ, MPQA, SST, TREC, MRPC)



# Ideas to understand
## QLora  https://www.youtube.com/watch?v=y9PHWGOa8HA
## SPQR



# open_source
* [[open assistant]]
* [tinygrad llama](https://github.com/tinygrad/tinygrad/blob/219a1f70630e12efe70b29c05184afe008e2e6d0/examples/llama.py#L313)
* [nano gpt](https://github.com/karpathy/nanoGPT/tree/master)
* [llama2](https://github.com/facebookresearch/llama/blob/7e1b864d574fe6f5ff75fa1d028feb269f7152d2/llama/model.py)
* [mistral](https://mistral.ai/news/announcing-mistral-7b/)

# My idea
# Tokenizer
Fixed tokenizer are wrong, use character based on build a hash for larger which have high confidence


# encoder/decoder
* Predict outcome encoding instead of distribution
* protect collapse of outcome vector and encoder by making them PCA middle thing from random encoder and decoder embeddings
* each level tries to predict next vector on that level instead of next token. So bottom layer next one token, second layer next 2, third layer next 4, fourth 8....
* and stop if can predict high confidence

# life experience
* weights are on edges of graph, and nodes have little color of what has passed through it.

# knowledge_base
* Read news find topic ask question based on news to existing resources if information does not exists add that information.
#ML
#LLM



# Embedding

## Meaning
For a next word prediction learning function:
* It is superposition of probabilities of next word.
	* Or superposition of a very compressed version /  class embedding of next word.
	* At this level exact next word is not possible.
* Which is also identity of a token, magically places similar tokens nearby.
* As we go up layers we get better estimate of probability of next word, as we see more things.

In last layer we find probability logits for embedding. This can be done in any layer too, but lesser accuracy, and faster model.
This is like embedding is superposition of basis vector and in last layer we are finding contribution of each basis vector.

Quest: Is this last layer basis vector true embedding of token vs, first layer embedding of token?
TODO: Can we run some embedding benchmarks?
https://arxiv.org/pdf/1608.05859.pdf: says same matrix can be used
https://openreview.net/pdf?id=xpFFI_NtgpW: Says not to do it, and higher dim in output than input.
https://openreview.net/pdf?id=H1eA7AEtvS: smaller input embedding, next sentence prediction.

Quest: Why does model not collapses to same embedding and output for all token?
Collapse is prevented as loss is surprise minimisation for actual next token. So in a collapsed case surprise is equal for all tokens, hence there is possibility to minimise it by moving away from collapsed state. Hence collapse is not a stable point.

### Problems:

What more can be embedding?
* Identity of embedding, an embedding should be able to identify itself, from other embedding.
* Previous word probabilities can also be part of embedding, which will improve it's identity.

arbitrary length embedding will change a lot depends upon where it finishes, as it is holding information of what is about to come, not what it is, and what is about to come depends on length of embedding.

Q learning:

Ideas
* Noise can be added to embedded (like dropout of neurons)
* they should grow in length with each layer.

Papers:
* Analyzing Transformers in Embedding Space
## Token

Length of token to embed:
* Too small a token length and can not identify what is next, as that is identity of embedding, hence will fail, too large a token length, will exponentially explode number of next token possible.
* In transformer token at any level is only concerned about mostly predicting it's next word, not about summarizing whole text.

### Properties of a good token
* Hierarchical in nature
	* It's identity should summarize everything in that token (sentence embedding), and can predict next/previous token at that hierarchy.

Improvement: we do not need whole decoding matrix. Given current encoding, how surprised I am if this encoding says next word is next word encoding, which can be anything arbitrarily assigned.
So we can chunk N character/N tokens to a new arbitrary token, such that previous chunk can predict this chunk.


## Position embedding
### Current approaches
* Learn position embedding.
* Add sin/cos frequency at start of model for position.
* Rotatory positional embedding
	* Not that wrong
	* Applies after Query and Key calculation, so tougher for model to use it.

### Properties of good position embedding
* Should be relative/local, not dependent on absolute position.
* Egoistic: Each token looks at rest of token, wrt itself.
* Token dependent: Token value can affect it's query radius, also a token value can affect it's signal to value radius.
* Non causal models support: Though most of models are causal, 

# Attention
## Theory
We want tokens to mix with other tokens
To do that we need N * N matrix of relationships of tokens
We also want to include positions of these tokens
* Given N vectors/tokens
* Compute Query, Key and Value for each token
* For each token compute it's Query * all Keys to find relation between tokens
* softmax of this for each token
* Take summation of values of all token in this proportion for each token.
# Wrong

## Half matrix
For rotatory embedding.
Use only bottom half of matrix as top half will be a copy of it.
This assumes a related to b is same as b related to a, since positions are added later.
Ideally looking back and looking forward should have implecations

## Fixed embedding
We need to have computation done in respect of each token. Each token can consider it self centre of universe and look at other tokens.

## 

# Ideas
## Flash Attention
* Better utilisation of gpu memory bandwidths.
* Bounded by N * N computation
* Bounded by memory loading in gpu, [Flash Attention]
## Grouped multi query attention
## Flash Attention
training and inference.

## KV Cache
Inference

### Attention vs feed forward
Given a token feed forward mixes dimension of token based on fixed mixer, for each token indepently.

Attention tries to mix a dimenstion across token for each dimension.
For a token same weighting is applied for each dimension across token computed dynamically once.

| attention | feed forward |
|---|---|
|mixes







## comments
* will soft max, be always positive, so can not have subtractive operation
	* So a query can be a negative query and have a corresponding negative value

# My ideas

Not all tokens should propagate till top, we can have some selection.

Completely parallelize computation, by increasing the scope of attention as level increase, and selecting only few tokens to propogate up.

## Logarithmic decomposition
Each token attends to log n tokens in past instead of n.
1 token with all history multi head
2 token with past half history multi head
3 token with past 1/4 tokens history,,,,



# Decoder
Currently it produces a distribution over output logits.

Instead we can produce the embedding vector of next element.

But this will collapse into single vector for all embeddings.


So each token gets a  random non learn embedding.
Then we will compress and decompress it to get back same random embedding.
And will use this comressed embedding as embedding in model.

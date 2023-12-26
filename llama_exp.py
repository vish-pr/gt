# Llama with experiments
# Positional embedding
# 1. No positional embedding only non cumulative merger so direction in inherent in the model
# 1.1 Maybe just sees last token
# 2 Positional embedding: depend on current word embedding compressed to few dmentsions and power from first token to statify sum =1

# 3 Each layer outpusts next token not just last layer
# 4 compression and decompression of text seen so far
# 5. Two way model, forward and backward each using same word embedding to predict word, for l to r forward and backward at any level are same ebedding.

# 6 Lora in iference. While inference change model to predict input, so burden of remembering context goes into model too.

# 7. Having bigger length vocab (multi token) too, and try to predit it too if it matches with low length vocab, and has high confidence, then use it.

# 8. when copying trained mode can have extra layer at top which are used by attention of future lower layers, but with much smaller context window, but they are not used for next word prediction, so are not focused on next wrord only, but on whole context.


# Exp encode and decoder key should be tranformable (linearly), i.e. ebedding and decoder layer for each token should be tranformable.

# Architecture:
# with increasing depth of transformer we should be able to predict longer tokens.
# there should be layer/branch parallel from each token whose responsibilty is not to predict next token so that it can focus on entire previous context.


# Exp 1 verification for embedding verify

from tinygrad.nn import Embedding, Linear


class Transformer:
  def __init__(self, dim: int, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_seq_len, ffn_dim_multiplier=None, n_kv_heads=None, dropout=0.0):
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)
    self.transform = Linear(dim, dim, bias=False)

  def __call__(self):

    # learn a linear transfomation



from sentencepiece import SentencePieceProcessor
from typing import List
import struct
import os
from pathlib import Path
from tinygrad.nn.state import torch_load, load_state_dict
from dataclasses import dataclass
from tinygrad.jit import TinyJit
from typing import Dict, Optional, Tuple
from tinygrad.shape.symbolic import Variable, sym_infer
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.helpers import getenv, dtypes, CI

JIT = getenv("JIT", 0 if CI else 1)


class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


@dataclass
class ModelArgs15M:
    # default hyperparameters for the Llama 7B model
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = 6
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.0

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1)*freqs.unsqueeze(dim=0)
  return Tensor.stack([Tensor.cos(freqs), Tensor.sin(freqs)], dim=-1).reshape(1, end, 1, dim//2, 2)


# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[
      1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == 5 and len(
      xk.shape) == 5 and len(freqs_cis.shape) == 5
  c, d = freqs_cis[:, :xq.shape[1], :, :,
                   0:1], freqs_cis[:, :xq.shape[1], :, :, 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
      return x
  return x[:, :, :, None, :].expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


class Attention:
  def __init__(self, dim, n_heads, n_kv_heads):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, cache_k: Optional[Tensor], cache_v: Optional[Tensor], start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor], jit_ctx: Optional[Dict[Variable, int]] = None) -> Tuple[Tensor, Tensor, Tensor]:
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert cache_k is not None and cache_v is not None, "no cache"
      assert start_pos == sym_infer(cache_k.shape[1], cache_k.lazydata.var_vals) == sym_infer(
          cache_v.shape[1], cache_v.lazydata.var_vals), f"cache has wrong shape, not ({start_pos} == {sym_infer(cache_k.shape[1], cache_k.lazydata.var_vals)} == {sym_infer(cache_v.shape[1], cache_v.lazydata.var_vals)})"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

    cache_k, cache_v = keys, values
    keys, values = repeat_kv(keys, self.n_rep).realize(
    ), repeat_kv(values, self.n_rep).realize()
    attn = Tensor.scaled_dot_product_attention(xq.transpose(1, 2), keys.transpose(
        1, 2), values.transpose(1, 2), mask).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn).realize(), cache_k.realize(), cache_v.realize()


class FeedForward:
  def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
    # TODO: what is this?
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))


class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, linear=Linear, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(
        dim, 4*dim, multiple_of, ffn_dim_multiplier)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, cache_k: Optional[Tensor], cache_v: Optional[Tensor], start_pos: int, freqs_cis: Tensor, mask: Optional[Tensor], jit_ctx: Optional[Dict[Variable, int]] = None):
    bsz, seqlen, _ = x.shape
    if JIT and mask is None:
      assert cache_k is not None and cache_v is not None, "no cache"
      pos = Variable("pos", 1, 1024)
      cache_k = cache_k.reshape(
          cache_k.shape[0], pos, cache_k.shape[2], cache_k.shape[3])
      cache_v = cache_v.reshape(
          cache_v.shape[0], pos, cache_v.shape[2], cache_v.shape[3])
      # need this because we don't reshape back to int shape in the jitted path and we don't have the correct var_vars in cache
      cache_k.lazydata.var_vals[pos] = start_pos
      cache_v.lazydata.var_vals[pos] = start_pos

    output, cache_k, cache_v = self.attention(self.attention_norm(
        x), cache_k, cache_v, start_pos, freqs_cis, mask, jit_ctx=jit_ctx)
    h = x + output
    return (h + self.feed_forward(self.ffn_norm(h))).realize(), cache_k.realize(), cache_v.realize()


class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_batch_size=32, max_seq_len=1024, ffn_dim_multiplier=None, n_kv_heads=None, rope_theta=10000, **kwargs):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, n_kv_heads,
                                    norm_eps, linear, ffn_dim_multiplier) for _ in range(n_layers)]
    self.kv_caches = [(None, None) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(
        dim // n_heads, max_seq_len * 2, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))

    self.tok_embeddings_jitted = TinyJit(
        lambda x: self.tok_embeddings(x).realize())
    self.postprocess_jitted = TinyJit(self.postprocess)
    self.layers_jitted = [TinyJit(layer.__call__) for layer in self.layers]

  def postprocess(self, x, temperature: Optional[float]):
    logits = self.output(self.norm(x))
    if temperature is not None:
        return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten().realize()
    return logits.realize()

  def __call__(self, tokens: Tensor, start_pos: int, temperature: Optional[float] = None):
    _bsz, seqlen = tokens.shape
    if seqlen == 1 and start_pos > 0 and JIT:
      pos = Variable("pos", 1, 1024)
      # get only the part of freqs_cis that we are using.
      freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (pos, pos+seqlen), (
          0, self.freqs_cis.shape[2]), (0, self.freqs_cis.shape[3]), (0, self.freqs_cis.shape[4])))
      freqs_cis.lazydata.var_vals[pos] = start_pos
      h = self.tok_embeddings_jitted(tokens)
      for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers_jitted, self.kv_caches)):
        h, cache_k, cache_v = layer(h, cache_k, cache_v, start_pos=start_pos,
                                    freqs_cis=freqs_cis, mask=None, jit_ctx={pos: start_pos})
        # TODO: move the kv cache into Attention, pre-allocate the cache and instead of cat, update the cache in-place
        self.kv_caches[i] = (cache_k, cache_v)
      return self.postprocess_jitted(h, temperature)
    else:
      freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (start_pos, start_pos+seqlen), (
          0, self.freqs_cis.shape[2]), (0, self.freqs_cis.shape[3]), (0, self.freqs_cis.shape[4])))
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"),
                         dtype=dtypes.float32).triu(start_pos+1).realize()
      h = self.tok_embeddings(tokens)
      for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers, self.kv_caches)):
        # need this reshape back to int shape in conversational mode because jitted and unjitted calls share the same cache
        if cache_k is not None and start_pos > 0:
          cache_k = cache_k.reshape(
              cache_k.shape[0], start_pos, cache_k.shape[2], cache_k.shape[3])
          cache_v = cache_v.reshape(
              cache_v.shape[0], start_pos, cache_v.shape[2], cache_v.shape[3])
        h, cache_k, cache_v = layer(
            h, cache_k, cache_v, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        self.kv_caches[i] = (cache_k, cache_v)
      return self.postprocess(h, temperature)


TOKENIZER_MODEL = "tokenizer.model"  # the llama sentencepiece tokenizer model


class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor()
        self.sp_model.LoadFromFile(model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.Encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.Decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            # sentencepiece uses this character as whitespace
            t = t.replace('▁', ' ')
            b = t.encode('utf-8')  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


if __name__ == '__main__':
  checkpoint_path = 'tiny_stories/weights/stories15M.pt'
  assert Path(checkpoint_path).is_file()
  check_point = torch_load(checkpoint_path)
  model_args: ModelArgs = check_point['model_args']
  print(check_point['config'])
  model = Transformer(**vars(ModelArgs15M()))
  weights = torch_load(checkpoint_path)['model']
  load_state_dict(model, weights, strict=False)
  tokenizer_path = 'tiny_stories/weights/tokenizer.model'
  assert Path(tokenizer_path).is_file()
  enc = Tokenizer(tokenizer_path)
  tokens = enc.encode('Ron had two cats,',  bos=True, eos=True)
  print(enc.decode(tokens))
  next = model(Tensor(tokens).unsqueeze(0), start_pos=0)
  print(next)

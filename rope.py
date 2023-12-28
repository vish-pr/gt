# rotatory position encoding
# https://arxiv.org/pdf/2104.09864v4.pdf
# implemetation matching https://github.com/facebookresearch/llama/blob/main/llama/model.py
from typing import Tuple
from tinygrad.helpers import dtypes

from tinygrad.tensor import Tensor


# def precompute_freqs_cis(head_dim: int, max_seq_len: int, rope_theta) -> Tuple[Tensor, Tensor]:
#   theta: float = rope_theta
#   # 1.0 / theta^((0, 2, 4, 6, ...)/dim)
#   assert head_dim % 2 == 0, "dim must be even else change below line to freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))"
#   freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2) / head_dim))
#   # [[0], [1], [2], [3], ..., [max_seq_len-1]] @ freqs as row vector ->  matrix of shape (max_seq_len, dim/2)
#   freqs = Tensor.arange(max_seq_len).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
#   freqs = freqs.reshape(1, max_seq_len, 1, head_dim // 2, 1)  # to match (batch_size, tokens, heads, head_dim//2, parity) parity is cos and sin
#   freqs.requires_grad = False
#   return Tensor.cos(freqs), Tensor.sin(freqs)


# def complex_mult(A: Tensor, c, d):
#   # (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
#   a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
#   ro = a * c - b * d
#   co = a * d + b * c
#   return ro.cat(co, dim=-1)


# def apply_rotary_emb_single_vector(vector: Tensor, freqs_cos, freqs_sin) -> Tensor:
#   assert freqs_cos.shape[1] >= vector.shape[1] and freqs_sin.shape[1] >= vector.shape[
#     1], f"freqs_cis shape mismatch {freqs_cos.shape} vector:{vector.shape}"
#   freqs_cos, freqs_sin = freqs_cos[:, :vector.shape[1], :, :], freqs_sin[:, :vector.shape[1], :, :]
#   assert vector.shape[-1] % 2 == 0, f"head_dim must be even, vector:{vector.shape}"
#   vector = vector.reshape(*vector.shape[0:-1], -1, 2)  # divides head_dim into 2 parts like [[0, 1], [2, 3], ...]
#   vector = complex_mult(vector, freqs_cos, freqs_sin)
#   return vector.flatten(3)  # merge divided 2 parts of head_dim

# # commentry: If we change which values of last dim to use for a pair (alternate vs first-half) will same pretrained model work?
# # Maybe no as we are mixing token differently which should result in different cosine similarities?


# def apply_rotary_emb(vectors: Tuple[Tensor, ...], freqs_cos, freqs_sin) -> Tuple[Tensor, ...]:
#   # return tuple of vector by applying rotary embedding to each vector in vectors
#   return tuple(apply_rotary_emb_single_vector(vector, freqs_cos, freqs_sin) for vector in vectors)


# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.default_float)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack([Tensor.cos(freqs), Tensor.sin(freqs)], dim=-1).reshape(1, end, 1, dim // 2, 2)


def complex_mult(A, c, d):
  # (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
  a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  ro = a * c - b * d
  co = a * d + b * c
  return ro.cat(co, dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[
    1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == 5 and len(xk.shape) == 5 and len(freqs_cis.shape) == 5
  c, d = freqs_cis[:, :xq.shape[1], :, :, 0:1], freqs_cis[:, :xq.shape[1], :, :, 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

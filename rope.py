# rotatory position encoding
# https://arxiv.org/pdf/2104.09864v4.pdf
# implemetation matching https://github.com/facebookresearch/llama/blob/main/llama/model.py
from typing import Tuple

from tinygrad.tensor import Tensor


def precompute_freqs_cis(head_dim: int, max_seq_len: int) -> Tuple[Tensor, Tensor]:
  theta: float = 10000.0
  # 1.0 / theta^((0, 2, 4, 6, ...)/dim)
  assert head_dim % 2 == 0, "dim must be even else change below line to freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))"
  freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2) / head_dim))
  # [[0], [1], [2], [3], ..., [max_seq_len-1]] @ freqs as row vector ->  matrix of shape (max_seq_len, dim/2)
  freqs = Tensor.arange(max_seq_len).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  freqs = freqs.reshape(1, max_seq_len, 1, head_dim // 2, 1)  # to match (batch_size, tokens, heads, head_dim//2, parity) parity is cos and sin
  freqs.requires_grad = False
  return Tensor.cos(freqs), Tensor.sin(freqs)


def complex_mult(A: Tensor, c, d):
  # (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
  a, b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  ro = a * c - b * d
  co = a * d + b * c
  return ro.cat(co, dim=-1)


def apply_rotary_emb_single_vector(vector: Tensor, freqs_cos, freqs_sin) -> Tensor:
  assert freqs_cos.shape[1] >= vector.shape[1] and freqs_sin.shape[1] >= vector.shape[
    1], f"freqs_cis shape mismatch {freqs_cos.shape} vector:{vector.shape}"
  freqs_cos, freqs_sin = freqs_cos[:, :vector.shape[1], :, :], freqs_sin[:, :vector.shape[1], :, :]
  assert vector.shape[-1] % 2 == 0, f"head_dim must be even, vector:{vector.shape}"
  vector = vector.reshape(*vector.shape[0:-1], -1, 2)  # divides head_dim into 2 parts like [[0, 1], [2, 3], ...]
  vector = complex_mult(vector, freqs_cos, freqs_sin)
  return vector.flatten(3)  # merge divided 2 parts of head_dim

# commentry: If we change which values of last dim to use for a pair (alternate vs first-half) will same pretrained model work?
# Maybe no as we are mixing token differently which should result in different cosine similarities?


def apply_rotary_emb(vectors: Tuple[Tensor, ...], freqs_cos, freqs_sin) -> Tuple[Tensor, ...]:
  # return tuple of vector by applying rotary embedding to each vector in vectors
  return tuple(apply_rotary_emb_single_vector(vector, freqs_cos, freqs_sin) for vector in vectors)

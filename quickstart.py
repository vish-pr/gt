# %%
# imports
import torch
# from tinygrad.helpers import DType, Timing, colored, fetch, getenv
from my_lm import Transformer
# from train.data_loader import DataLoader
from tinygrad import Device, Tensor, dtypes, nn, mlops
from tinygrad.helpers import Timing
import numpy as np
# from tinygrad.helpers import dtypes
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.optim import SGD
from extra.datasets import fetch_mnist
from sentencepiece import SentencePieceProcessor

# %%
# selector
def argmax(x, keepdim=False):
  axis = 1
  print('max', x.shape, x.max(axis=axis, keepdim=keepdim).numpy())
  m = x > (x.max(axis=axis, keepdim=True) *.8)
  print('m', m.numpy(), (m* Tensor.arange(x.shape[axis] -1, -1, -1, requires_grad=False, device=x.device).reshape(x.shape[axis], *[1]*(x.ndim-axis-1))).numpy())
  idx = m * Tensor.arange(x.shape[axis]-1,-1,-1, requires_grad=False, device=x.device).reshape(x.shape[axis], *[1]*(x.ndim-axis-1))
  print('idx', idx.max(axis=axis, keepdim=keepdim).numpy())
  return (x.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1).cast(dtypes.default_int)
a = Tensor([[.2, 3, 8.8, 9, 7.99, 1.89, 8.9],[.2, 3.9, 4, .5, .9, 1.89, .8]]).softmax(-1)
# topk algorithm
mask = Tensor.ones_like(a)
for i in range(3):
  max = a.max(-1)
  index = argmax(a, keepdim=True)
  mask[index] = 0
  print(index.numpy(), mask.numpy())
max = a.max(-1)
min = max * .9
c = a[a > min]
print(c.numpy())
b = (a > min).where(a, Tensor(0))
print(b.numpy())
print(a.numpy(), max.numpy())

# %%
# attention mask
attn_mask = Tensor.zeros(3, 13, requires_grad=False).triu(1).cast(dtypes.bool)
mask = Tensor.full((1, 1, 3, 13), float("-inf")).triu(11).realize()
print(attn_mask.numpy())
b = attn_mask[:,:0]
c = Tensor([[True], [False], [True]])
d = b.cat(c, dim=1)
print(d.shape, d.numpy())
print(mask.numpy())


# %%
# bfloat
from tinygrad.tensor import Tensor
url = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin'
filename = "weights/" + "mistral-7b-v01" + '-' + url.split("/")[-1]
# b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.float16).realize()
# print(b)
# b = b.bitcast(dtypes.uint16).realize()
# print(b.numpy())
# b = b.bitcast(dtypes.float16).realize()
# print(b.numpy())

weights = nn.state.torch_load(str(fetch(url, filename)))

a: Tensor = weights['model.embed_tokens.weight']
print(a)
a = a.bitcast(dtypes.uint16).realize().to('CPU').realize()
a = a.cast(dtypes.uint32).mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).cast(dtype=dtypes.float16).realize()
# a = a.to('CPU').cast(dtype=dtypes.float16).realize()
# .cast(dtypes.uint32).mul(1 << 16).realize()  # contiguous().bitcast(dtypes.float32).cast(dtype)
# a = a.cast(dtype=dtypes.float16).realize()
# a = a.to('CPU').realize()
print(a.numpy())
print(weights)
a = weights['model.layers.1.input_layernorm.weight']

shape = a.shape
a = a.to(Device.DEFAULT).realize()
print(a)
a = mlops.Cast.apply(a, dtype=dtypes.uint64, bitcast=True).realize()
print(a)
print(a.numpy())
print((a.sum() > 1).realize().numpy())


# %%

# %%
# load model from file on disk
model = torch.load('weights/mistral-7b-v01-pytorch_model-00001-of-00002.bin')

# %%.
# pytoch bitcast
print(model['model.embed_tokens.weight'].type(torch.float16).numpy())

 %%
# rope
dim = 10
theta = 10000
max_seq_len = 16
freqs = Tensor.arange(0, dim, 2)[:(dim // 2)]
freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
# [[0], [1], [2], [3], ..., [max_seq_len-1]] @ freqs as row vector ->  matrix of shape (max_seq_len, dim/2)
freqs = Tensor.arange(max_seq_len).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
freqs.reshape(1, max_seq_len, 1, dim // 2)
print(freqs.numpy())
x = Tensor.arange(16)
print(x.reshape(-1, 2).numpy())


# %%
# other

class TinyJit:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("Before calling the function")
        result = self.func(*args, **kwargs)
        print("After calling the function")
        return result


@TinyJit
def my_function():
    print("Inside the function")


my_function()


tokenizer = SentencePieceProcessor()
tokenizer_path = 'tiny_stories/weights/tok512.model'
tokenizer.LoadFromFile(tokenizer_path)
data_loader = DataLoader('tiny_stories/data/TinyStoriesV2-GPT4-train.txt',
                         'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt', 1, tokenizer)
vocab_size = tokenizer.vocab_size()


class TinyNet:
  def __init__(self, dim=164):
    self.tok_embeddings = Embedding(vocab_size, dim)
    # self.weights = Tensor.uniform(64, 64)
    self.output = Linear(dim, vocab_size, bias=False)

  def __call__(self, x):
    x = self.tok_embeddings(x)
    x = self.output(x)
    return x


net = Transformer(64, vocab_size)


opt = SGD([net.tok_embeddings.weight, net.output.weight], lr=3e-4)

# X_train, Y_train, X_test, Y_test = fetch_mnist()

with Tensor.train():
  for step in range(1000):
    # random sample a batch
    # samp = np.random.randint(0, X_train.shape[0], size=(64))
    # batch = Tensor(X_train[samp], requires_grad=False)
    # # get the corresponding labels
    # labels = Tensor(Y_train[samp])
    batch = data_loader.get_batch(False)

    # forward pass
    out = net(batch)

    # compute loss
    loss = out.sparse_categorical_crossentropy(batch)

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = out.argmax(axis=-1)
    acc = (pred == batch).mean()

    if step % 100 == 0:
      print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

with Timing("Time: "):
  avg_acc = 0
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / 1000}")

# %%

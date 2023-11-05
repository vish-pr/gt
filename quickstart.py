# %%
# imports
from my_lm import Transformer
from data_loader import DataLoader
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing
import numpy as np
from tinygrad.helpers import dtypes
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.optim import SGD
from extra.datasets import fetch_mnist
from sentencepiece import SentencePieceProcessor


# %%
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

# Plots the stats dumped by the model.
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from train.model import TensorStats

file_name = 'tiny_stories/weights/my_stories_dim64_layer5_ckpt25_model_stats_dump.csv'

# read a csv file of format: name, iter, mean standard deviation, min, max

stats: Dict[str, List[TensorStats]] = {}

with open(file_name, 'r') as f:
  while (line := f.readline()):
    line = line.split(',')
    name = line[0]
    iter = int(line[1])  # cast iter to int
    mean = float(line[2])  # cast mean to float
    std = float(line[3])  # cast std to float
    min = float(line[4])  # cast min to float
    max = float(line[5])  # cast max to float
    if 'counter' in name:
      continue
    # add to stats
    if stats.get(name) is None:
      stats[name] = [TensorStats(name, iter, mean, std, min, max)]
    else:
      stats[name].append(TensorStats(name, iter, mean, std, min, max))

# plot the stats


def name_grouper(name) -> int:
  # if name ends in wq.weright, return 0
  if 'wq.weight' in name:
    return 0
  elif 'wk.weight' in name:
    return 1
  elif 'wv.weight' in name:
    return 2
  elif 'wo.weight' in name:
    return 3
  elif 'feed_forward' in name:
    return 4
  elif 'output' in name:
    return 5
  else:
    return 6


fig, axs = plt.subplots(7, 2, sharex=True)

for name, stats_list in stats.items():
  if 'norm.weight' in name or 'freq_' in name:
    continue
  iter = np.array([stat.iter for stat in stats_list])
  mean = np.array([stat.mean for stat in stats_list])
  std = np.array([stat.std for stat in stats_list])  # Extract std values
  max_val = np.array([stat.max for stat in stats_list])  # Extract max values
  min_val = np.array([stat.min for stat in stats_list])  # Extract min values

  i = name_grouper(name)
  axs[i][0 if 'grad' in name else 1].plot(iter, mean, label=name)
  axs[i][0 if 'grad' in name else 1].legend()
  axs[i][0 if 'grad' in name else 1].fill_between(iter, mean - std, mean + std, alpha=0.05)  # Plot std as filled area
  # axs[i][0 if 'grad' in name else 1].plot(iter, max_val, label=name + '_max', linestyle='dashed')  # Plot max values
  # axs[i][0 if 'grad' in name else 1].plot(iter, min_val, label=name + '_min', linestyle='dashed')  # Plot min values

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.show()

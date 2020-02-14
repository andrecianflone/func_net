import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

class NpDataset(Dataset):
  def __init__(self, array):
    self.array = array
  def __len__(self): return len(self.array)
  def __add__(self, ds):
        return data_utils.TensorDataset(  torch.from_numpy(self.array + ds))

  def __getitem__(self, i): return self.array[i]

def generate_input(low,high,samples, length, batch_size):
    """
    Generate input of size
    """
    x = np.random.randint(low, high, (samples, length))
    data_x = NpDataset(x)
    d1 = DataLoader(x, batch_size=batch_size)

    return d1

def operators(data, op):
    """
    Given some data and operator id, will apply an operation
    """

if __name__ == '__main__':
    # Debug data
    batch_size = 5
    samples = 100
    seq_len = 10
    low=0
    high=10

    d1 = generate_input(low, high, samples, seq_len, batch_size)

    pass

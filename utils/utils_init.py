import numpy as np
import random
import os
import torch

def init_seed(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash
    random.seed(seed) # Python
    np.random.seed(seed) # NumPy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # PyTorch GPU
    # CuDNN
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    # torch.use_deterministic_algorithms(True)
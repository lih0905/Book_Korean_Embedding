import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dev

sns.set()
plt.rcParams["font.family"] = 'NanumBarunGothic'


print('hello, world!')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

from tqdm import tqdm


INPUT_SIZE = 11
OUTPUT_SIZE = 11
SEED = 0

#Hyperparameters
learning_rate = 5e-2
epochs = 5
hidded_layer_size = 64
attention_layers_count = 2
act_fn = F.relu
num_heads = 4


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, hidded_layer_size)
        self.fc2 = nn.Linear(hidded_layer_size, OUTPUT_SIZE)
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(hidded_layer_size, num_heads) for i in range(attention_layers_count)]
        )
        self.q = nn.ModuleList([nn.Linear(hidded_layer_size, hidded_layer_size) for i in range(attention_layers_count)])
        self.k = nn.ModuleList([nn.Linear(hidded_layer_size, hidded_layer_size) for i in range(attention_layers_count)])
        self.v = nn.ModuleList([nn.Linear(hidded_layer_size, hidded_layer_size) for i in range(attention_layers_count)])
        self.attention_layers_count = attention_layers_count

    def forward(self, X):
        X = act_fn(self.fc1(X))
        for i in range(self.attention_layers_count):
            X = act_fn(self.attentions[i](self.q[i](X), self.k[i](X), self.v[i](X), need_weights=False)[0])
        X = act_fn(self.fc2(X))
        X = F.softmax(X, dim=1)
        return X

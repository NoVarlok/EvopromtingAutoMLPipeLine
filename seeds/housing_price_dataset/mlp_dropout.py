import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

from tqdm import tqdm


INPUT_SIZE = 7
OUTPUT_SIZE = 1
SEED = 0

#Hyperparameters
learning_rate = 1e-2
epochs = 10
hidded_layer = 128
layers_count = 3
act_fn = F.leaky_relu
dropout_rate = 0.05


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
        layers_inputs = [hidded_layer for _ in range(layers_count)]
        layers_outputs = [hidded_layer for _ in range(layers_count)]
        layers_inputs[0] = INPUT_SIZE
        layers_outputs[-1] = OUTPUT_SIZE
        self.layers = nn.ModuleList([nn.Linear(input_size, output_size) for input_size, output_size in zip(layers_inputs, layers_outputs)])
        self.layers_count = layers_count

    def forward(self, X):
        for i in range(self.layers_count - 1):
            X = act_fn(F.dropout(self.layers[i](X), dropout_rate))
        X = self.layers[-1](X)
        return X


def main(train_dataset, test_dataset, metric_fn, loss_fn, device):
    model = Model()
    model_paramters_count = count_parameters(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metric = 0
    batch_count = 0

    for epoch in range(epochs):
        for X, y in train_dataset:
            X = X.to(device)
            y = torch.unsqueeze(y, 1).to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        for X, y in test_dataset:
            X = X.to(device)
            y = torch.unsqueeze(y, 1).to(device)
            output = model(X)
            metric += metric_fn(y, output)
            batch_count += 1

    metric /= batch_count

    return float(metric), int(model_paramters_count)

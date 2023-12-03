import concurrent.futures
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import openai

# from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import torch.nn.functional as F

# from seeds.mlp import main
from seeds.mlp_skip_connections import main
# from seeds.mlp_dropout import main


def prepare_data_tensor(csv_path, target_name, batch_size):
    train = pd.read_csv(csv_path)
    train_target = torch.tensor(train[target_name].values.astype(np.float32))
    train = torch.tensor(train.drop(target_name, axis = 1).values.astype(np.float32)) 
    train_tensor = data_utils.TensorDataset(train, train_target) 
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
    return train_loader


if __name__ == '__main__':
    target_name = 'Price'
    train_csv_path = '/home/lyakhtin/repos/hse/krylov2/PrepareDatasets/prepared_datasets/housing_price_dataset_processed/train.csv'
    test_csv_path = '/home/lyakhtin/repos/hse/krylov2/PrepareDatasets/prepared_datasets/housing_price_dataset_processed/test.csv'

    batch_size = 1000
    train_dataloader = prepare_data_tensor(train_csv_path, target_name, batch_size)
    test_dataloader = prepare_data_tensor(test_csv_path, target_name, batch_size)

    metric_fn = F.mse_loss
    loss_fn = F.mse_loss
    device = 'cuda:0'

    metric, model_paramters_count = main(train_dataloader, test_dataloader, metric_fn, loss_fn, device)
    print('Metric:', metric)
    print('Params:', model_paramters_count)

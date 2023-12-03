import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_path, target_name):
        df = pd.read_csv(csv_path)
        self.X_tensor = torch.from_numpy(df.drop(target_name, axis=1).values).float()
        self.y_tensor = torch.from_numpy(df[target].values).float()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]

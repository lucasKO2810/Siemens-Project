import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class NewDataset(Dataset):
    def __init__(self, file_name):
        file = pd.read_excel(file_name)
        x = file.iloc[0:1001, 1:3].values
        y = file.iloc[0:1001, 3].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

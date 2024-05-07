import pandas as pd
import torch
from torch.utils.data import Dataset


class Color2NormalDataset(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv)
    def __len__(self):
        return self.data['X'].count()

    def __getitem__(self,idx):
        x = self.data['X'][idx]/120
        y = self.data['Y'][idx]/160
        r = self.data['R'][idx]/255
        g = self.data['G'][idx]/255
        b = self.data['B'][idx]/255
        nx = self.data['Nx'][idx]/255
        ny = self.data['Ny'][idx]/255
        nz= self.data['Nz'][idx]/255
        return torch.tensor((x, y, nx, ny, nz), dtype=torch.float32), torch.tensor((r,g,b), dtype=torch.float32)

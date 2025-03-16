import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ToxinDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float64)
        self.targets = torch.tensor(y, dtype=torch.float64)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getdim__(self):
        return self.features.shape

class ToxinMLPRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.input_layer = nn.Linear(num_features, 16)
        self.hidden_layer1 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x
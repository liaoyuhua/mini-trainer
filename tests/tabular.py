import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

POLY_DEGREE = 4
SAMPLE_SIZE = 500
BATCH_SIZE = 32

W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(POLY_DEGREE, POLY_DEGREE * 2)
        self.fc2 = torch.nn.Linear(POLY_DEGREE * 2, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TabularDataset(Dataset):
    def __init__(self):
        random = torch.randn(SAMPLE_SIZE)
        self.x = make_features(random)
        self.y = f(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_loader = DataLoader(TabularDataset(), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TabularDataset(), batch_size=BATCH_SIZE, shuffle=False)

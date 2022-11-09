import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mini.trainer import Trainer

df = pd.read_csv('./data/houseprice.csv', usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                        "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna().reset_index(drop=True)


# encoding
cats = ["MSZoning", "Street", "LotShape"]
for c in cats:
    enc = LabelEncoder()
    df[c] = enc.fit_transform(df[c].values)

# split train, validation, test dataset
traindata, valdata, testdata = df.iloc[:int(len(df)*0.8)], df.iloc[int(len(df)*0.8):int(len(df)*0.9)], df.iloc[int(len(df)*0.9):]

# pytorch dataset, dataloader
class PriceDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = torch.tensor(x.values, dtype=torch.float)
        self.y = torch.tensor(y.values, dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

trainset = PriceDataset(traindata.drop("SalePrice", axis=1), traindata["SalePrice"])
valset = PriceDataset(valdata.drop("SalePrice", axis=1), valdata["SalePrice"])
testset = PriceDataset(testdata.drop("SalePrice", axis=1), testdata["SalePrice"])

trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
validloader = DataLoader(valset, batch_size=200, shuffle=False)
testloader = DataLoader(testset, batch_size=200, shuffle=False)


# network
class DNN(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.out(self.fc(x)).squeeze()

model = DNN(9, 18)

trainer = Trainer(model, save_path="./", lr=0.01)

trainer.fit(trainloader, validloader, epochs=20, prog_bar=True)

pred = trainer.predict(testloader)

trainer.plot_loss()

trainer.log()
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader



class weatherdata(Dataset):
    def __init__(self):
        # Training Set
        df = pd.read_csv("train_set.csv")
        self.X = torch.tensor(df.iloc[:, 1:-1].to_numpy(), dtype=torch.float)
        self.y = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.float)
        self.len = len(df)

        # Validation Set
        df_valid = pd.read_csv("validate_set.csv")
        self.X_valid = torch.tensor(df_valid.iloc[:, 1:-1].to_numpy(), dtype=torch.float)
        self.y_valid = torch.tensor(df_valid.iloc[:, -1].to_numpy(), dtype=torch.float)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class forcast(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(forcast, self).__init__()

        self.in_to_h1 = nn.Linear(10, 5)
        self.h1_to_out = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        return self.h1_to_out(x)


def trainNN(epochs=5, batch_size=32, lr=0.001):
    # Create the dataset
    wd = weatherdata()

    # Create data loader
    data_loader = DataLoader(wd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create an instance of the Neural Network
    fc = forcast()

    # Mean Square Error loss function
    mse_loss = nn.MSELoss(reduction='sum')

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(fc.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(data_loader, 0):
            x, y = data

            optimizer.zero_grad()

            output = fc(x)

            loss = mse_loss(output.view(-1), y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch} of {epochs} MSE (Train): {running_loss / len(wd)}")
        running_loss = 0.0
        with torch.no_grad():
            output = fc(wd.X_valid).view(-1)
        print(f"Epoch {epoch} of {epochs} MSE (Validation): {torch.mean((output - wd.y_valid) ** 2.0)}")
        print("-" * 50)
    return fc


# Both BaggingClassifier and RandomForestClassifier
# Has been used on the dataset before
# BaggingClassifier MSE: 0.518
# ForestClassifier MSE: 0.372
trainNN(epochs=50)

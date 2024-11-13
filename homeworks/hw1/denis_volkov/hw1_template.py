import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix

class MLP(nn.Module):
    def __init__(self, input_size: int, classes: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc3 = nn.Linear(25, classes)

    def forward(self, x: torch.Tensor):
        out1 = torch.relu(self.bn1(self.fc1(x)))
        out2 = torch.relu(self.bn2(self.fc2(out1)))
        out3 = self.fc3(out2)
        return out3
    
    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.forward(x) 
            probabilities = torch.softmax(x, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes


def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    x_columuns = ["order0", "order1", "order2"]
    y_column = "order0"

    X_train = train_df.drop(x_columuns, axis=1).values
    y_train = train_df[y_column].values

    X_val = val_df.drop(x_columuns, axis=1).values
    y_val = val_df[y_column].values

    X_test = test_df.values
    return (X_train, y_train, X_val, y_val, X_test)


def init_model(device: torch.device, learning_rate: float, input_size: int, classes: int):
    model = MLP(input_size, classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def evaluate(model: MLP, X: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X).cpu().numpy()
        y_np = y.cpu().numpy()
        
        accuracy = accuracy_score(y_np, predictions)
        conf_matrix = confusion_matrix(y_np, predictions)
        return predictions, accuracy, conf_matrix


def train(model: MLP, 
          criterion: nn.CrossEntropyLoss, 
          optimizer: optim.Optimizer, 
          X_train: torch.Tensor, y_train: torch.Tensor, 
          X_val: torch.Tensor, y_val: torch.Tensor, 
          batch_size: int, epochs: int
    ):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            optimizer.zero_grad()

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= X_train.size(0)
        val_predictions, val_accuracy, val_conf_matrix = evaluate(model, X_val, y_val)
        print(f'Epoch {epoch}, Loss {train_loss}, Val. Accuracy {val_accuracy}')

    return model

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def main(args):
    device = get_device()

    X_train, y_train, X_val, y_val, X_test = load_data(
        args.train_csv, 
        args.val_csv, 
        args.test_csv
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    model, criterion, optimizer = init_model(
        device, 
        learning_rate=args.lr, 
        input_size=X_train.shape[1],
        classes=3
    )
    

    train(
        model, 
        criterion, 
        optimizer, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        args.batch_size, 
        args.num_epochs
    )

    
    y_test = model.predict(X_test).cpu().numpy()
    pd.Series(y_test).to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/data/submission.csv')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=1024)
    parser.add_argument('--num_epochs', default=50)

    args = parser.parse_args()
    main(args)

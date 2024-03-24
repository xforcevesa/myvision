import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.nn import functional as F
import tqdm
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = 'cpu'


def process_data(x, y):
    x = np.asanyarray(x).copy()
    x = torch.from_numpy(x).float().to(device)
    y = F.one_hot(torch.asarray(y), num_classes=10).float().to(device)
    return [x, y]


def train_model(model: nn.Module, batch_size: int, learning_rate: float, epochs: int, train_rate: float):
    ts = transforms.Compose([
        transforms.PILToTensor()
    ])
    dataset = datasets.MNIST(
        root='./dataset',
        download=True,
        transform=ts
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset, valid_dataset = data.random_split(dataset, [train_rate, (1 - train_rate)])

    train_dataset = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = data.DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):
        train_loss = 0.
        # train_progress_bar = tqdm.tqdm(range(len(train_dataset)))
        for x, y in train_dataset:
            optimizer.zero_grad()
            x, y = process_data(x, y)
            output = model(x)
            loss = criterion(output, y)
            train_loss += loss.detach().cpu().numpy().mean()
            loss.backward()
            optimizer.step()
            # train_progress_bar.update(1)
        train_loss /= len(train_dataset)

        valid_loss = 0.
        accuracy = 0.
        # valid_progress_bar = tqdm.tqdm(range(len(valid_dataset)))
        for x, y in valid_dataset:
            x, y = process_data(x, y)
            output = model(x)
            loss = criterion(output, y)
            loss = loss.detach().cpu().numpy()
            valid_loss += loss.mean()
            # print((output.argmax(axis=1) == y.argmax(axis=1)).sum())
            accuracy += (output.argmax(axis=1) == y.argmax(axis=1)).sum()
            # valid_progress_bar.update(1)
        valid_loss /= len(valid_dataset)
        accuracy /= len(valid_dataset) * batch_size

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")


def main():
    from models import mlp
    train_model(
        model=mlp.MLP(
            input_dim=28 * 28,
            output_dim=10,
            hidden_dim=300,
            num_layers=3
        ).to(device),
        batch_size=2400,
        learning_rate=0.001,
        epochs=100,
        train_rate=0.7
    )


if __name__ == '__main__':
    main()

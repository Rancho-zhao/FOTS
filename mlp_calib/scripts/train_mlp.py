import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train import MLP, Color2NormalDataset

seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def train(train_loader, epochs, lr):
    model = MLP().to(device)
    wandb.init(project="MLP", name="Normal 2 Color model train")
    wandb.watch(model, log_freq=100)

    model.train()

    learning_rate = lr
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = epochs
    avg_loss=0.0
    loss_record=[]
    cnt=0
    total_step = len(train_loader)
    for epoch in tqdm(range(1, 1 + num_epochs)):
        for i, (data, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt+=1

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                loss_record.append(loss.item())
                # wandb.log({"Mini-batch loss": loss})
        # wandb.log({'Running test loss': avg_loss / cnt})
    os.makedirs(f"{base_path}/models", exist_ok=True)
    print(f"Saving model to {base_path}/models/")
    # torch.save(model,
    #            f"{base_path}/models/mlp_n2c_r.ckpt")
    torch.save(model.state_dict(), f"{base_path}/models/mlp_n2c_gelsight1.pth")


def test(test_loader,criterion):
    model = MLP().to(device)
    model.load_state_dict(torch.load(f"{base_path}/models/mlp_n2c_gelsight1.pth"))
    model.to(device)
    # model = torch.load(
    #     f"{base_path}/models/mlp_n2c_r.ckpt").to(
    #     device)
    model.eval()
    wandb.init(project="MLP", name="Normal 2 Color model test")
    wandb.watch(model, log_freq=100)
    model.eval()
    avg_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            cnt=cnt+1
            # wandb.log({"Mini-batch test loss": loss})
        avg_loss = avg_loss / cnt
        print("Test loss: {:.4f}".format(avg_loss))
        # wandb.log({'Average Test loss': avg_loss})


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    argparser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    argparser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    argparser.add_argument('--epochs', type=int, default=200, help='epochs')
    argparser.add_argument('--train_path', type=str, default=f'{base_path}/datasets/train_test_split/non_zeros.csv',
                           help='data path')
    argparser.add_argument('--test_path', type=str, default=f'{base_path}/datasets/train_test_split/test.csv',
                           help='test data path')
    option = argparser.parse_args()

    if option.mode == "train":
        train_set = Color2NormalDataset(
            option.train_path)
        train_loader = DataLoader(train_set, batch_size=option.batch_size, shuffle=True)
        print("Training set size: ", len(train_set))
        train(train_loader, option.epochs,option.learning_rate)
    elif option.mode == "test":
        test_set = Color2NormalDataset(
            option.test_path)
        test_loader = DataLoader(test_set, batch_size=option.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        test(test_loader, criterion)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from dataclasses import dataclass, field
from .digit_dataset import CustomDigitDataset
import yaml
from typing import List
import argparse
from . import utils


class Net(nn.Module):
    def __init__(self, letters=['0','1','2','3','4','5','6','7','8','9','.']):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(15488, 256)
        self.fc2 = nn.Linear(256, len(letters))
        self.letters = letters

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@dataclass
class MNIST_Args(yaml.YAMLObject):
    batch_size : int = 10
    test_batch_size : int = 10
    gamma : float = 0.1
    lr : float = 0.1
    epochs : int = 5
    save_model : bool = True
    log_interval : int = 50
    dry_run : bool = False
    use_cuda: bool = torch.cuda.is_available()
    model_checkpoint: str = "mnist_cnn.pt"
    train_size: int = 15000
    test_size: int = 5000
    letters: List[str] = field(default_factory=lambda: ['0','1','2','3','4','5','6','7','8','9','.'])


def train_loop(args: MNIST_Args, device):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if args.use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = CustomDigitDataset(args.train_size, letters=args.letters)
    dataset2 = CustomDigitDataset(args.test_size, letters=args.letters)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(args.letters).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), args.model_checkpoint)


def load_model(device, model_checkpoint=utils.DATA_FOLDER+"/mnist_cnn.pt"):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Digit Model Training',
        description='Recognizes 28x28 digits',
        epilog='')
    parser.add_argument("config")
    cmd_args = parser.parse_args()

    with open(cmd_args.config) as stream:
        try:
            config = yaml.safe_load(stream)
            args = MNIST_Args(**config)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    train_loop(args, utils.select_device())


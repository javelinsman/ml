import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


def fetch_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=2
    )
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=8, shuffle=True, num_workers=2
    )
    return train_loader, test_loader


class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, 7)
    self.conv2 = nn.Conv2d(16, 4, 5)
    self.pool = nn.MaxPool2d(2, 2)
    
    self.deconv1 = nn.ConvTranspose2d(4, 8, 5)
    self.deconv2 = nn.ConvTranspose2d(8, 16, 10)
    self.deconv3 = nn.ConvTranspose2d(16, 1, 13)
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.deconv1(x))
    x = F.relu(self.deconv2(x))
    x = self.deconv3(x).clamp(0, 1)
    return x


def train(model, train_loader, epochs=50, print_every=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_iter, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_iter % print_every == 0:
                print(f'[{epoch + 1}, {batch_iter}] loss: {running_loss / print_every}')
                running_loss = 0.0
    print('Finished training')

def test(model, images):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        return outputs.cpu()
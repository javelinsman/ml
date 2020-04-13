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


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 512)
        self.fc_mu = nn.Linear(512, 32)
        self.fc_log_var = nn.Linear(512, 32)
        self.fc_decode1 = nn.Linear(32, 512)
        self.fc_decode2 = nn.Linear(512, 784)

    def encode(self, x):
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_log_var(x)
        z_std = torch.exp(0.5 * z_log_var)
        eps = torch.zeros_like(z_std).normal_()
        z = z_mu + z_std * eps
        # z = torch.normal(z_mu, std) <- NEVER DO THIS!
        return z_mu, z_log_var, z

    def decode(self, z):
        z = F.relu(self.fc_decode1(z))
        z = torch.sigmoid(self.fc_decode2(z))
        return z.view(z.size(0), 1, 28, 28)
        
    def forward(self, x):
        z_mu, z_log_var, z = self.encode(x)
        return self.decode(z)

def train(model, train_loader, epochs=50, print_every=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0
        for batch_iter, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            z_mu, z_log_var, z = model.encode(inputs)
            outputs = model.decode(z)

            recon_loss = criterion(outputs, inputs)
            kl_loss = 1 + z_log_var - z_mu ** 2 - torch.exp(z_log_var)
            kl_loss = torch.sum(kl_loss) * -0.5
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kl_loss.item()

            if batch_iter % print_every == 0:
                print(f'[{epoch + 1}, {batch_iter}] loss: {running_loss / print_every}, recon {running_recon / print_every}, kld {running_kld / print_every}')
                running_recon = 0.0
                running_kld = 0.0
                running_loss = 0.0
    print('Finished training')

def test(model, images):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        return outputs.cpu()
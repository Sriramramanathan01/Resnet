import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch import cuda
import datetime

"""
Loading the dataset
"""
path = "./Datasets"
cifar10 = datasets.CIFAR10(path, train = True, download = True, transform = transforms.Compose([transforms.RandomCrop(size=[32,32], padding=4),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                                                     std=(0.2470, 0.2435, 0.2616))
                                                                                                ]))

cifar10_val = datasets.CIFAR10(path, train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

"""
Network Model
"""
"""
Residual Block
"""
class ResBlock(nn.Module):
  def __init__(self, n_chans):
    super(ResBlock, self).__init__()
    self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
    self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
    torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
    torch.nn.init.constant_(self.batch_norm.weight, 0.5)
    torch.nn.init.zeros_(self.batch_norm.bias)

  def forward(self, x):
    out = self.conv(x)
    out = self.batch_norm(out)
    out = torch.relu(out)
    return out + x

"""  
ResNet
"""
class DeepResNet(nn.Module):
  def __init__(self, n_chans1=32, n_blocks = 6):
    super().__init__()
    self.n_chans1 = n_chans1
    self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(n_chans1, n_chans1 * 2, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(n_chans1 * 2, n_chans1 * 4, kernel_size=3, padding=1)
    self.resblocks_1 = nn.Sequential(*(n_blocks * [ResBlock(n_chans = n_chans1)]))
    self.resblocks_2 = nn.Sequential(*(n_blocks * [ResBlock(n_chans = n_chans1 * 2)]))
    self.resblocks_3 = nn.Sequential(*(n_blocks * [ResBlock(n_chans = n_chans1 * 4)]))
    self.fc1 = nn.Linear(8 * 8 * n_chans1*4, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    out = torch.relu(self.conv1(x))
    out = self.resblocks_1(out)
    out = F.max_pool2d(out, 2)
    out = self.conv2(out)
    out = self.resblocks_2(out)
    out = F.max_pool2d(out, 2)
    out = self.conv3(out)
    out = self.resblocks_3(out)
    out = out.view(-1, 8 * 8 * self.n_chans1*4)
    out = torch.relu(self.fc1(out))
    out = self.fc2(out)
    return out

  
"""
Training Loop
"""
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
  for epoch in range(1, n_epochs+1):
    loss_train = 0.0

    for imgs, labels in train_loader:
      imgs = imgs.to(device = device)
      labels = labels.to(device = device)
      outputs = model(imgs)
      loss = loss_fn(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_train += loss.item()

    print("{}, Epoch: {}, Training loss: {}".format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    
    
train_loader = torch.utils.data.DataLoader(cifar10, batch_size=100, shuffle=True)
model = DeepResNet().to(device = device)
optimizer = optim.SGD(model.parameters(), lr = 1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 50,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader)

"""
Validation
"""
train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=False)

def validate(model, train_loader, val_loader):
  for name, loader in [("train", train_loader), ("val", val_loader)]:
    correct = 0
    total = 0

    with torch.no_grad():
      for imgs, labels in loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

    print("Accuracy {}: {:.2f}".format(name, correct/total))

validate(model, train_loader, val_loader)

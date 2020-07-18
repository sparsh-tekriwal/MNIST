!pip install torchvision
!pip install torch

import torchvision, torch
from torchvision import transforms, datasets

# storing files in the same folder and transforming to Tensors
train = datasets.MNIST("", train=True, download=True, transform= transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform= transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import matplotlib.pyplot as plt
plt.imshow(data[0][8].view(28,28))
plt.show()

value_counts = {}
for data in trainset:
  for y in data[1]:
    if int(y) in dic.keys():
      value_counts[int(y)] +=1
    else:
      value_counts[int(y)] =1
print(value_counts)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.softmax(self.fc4(x), dim=1)

    return x

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
  for data in trainset:
    X,y = data
    net.zero_grad()
    output = net(X.view(-1, 28*28))
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
  print(loss)


correct = 0
total = 0

with torch.no_grad():
  for data in trainset:
    X,y = data
    output = net(X.view(-1,784))
    for idx, i in enumerate(output):
      if torch.argmax(i) == y[idx]:
        correct += 1
      total +=1
print(" Training accuracy: ", round(correct/total, 3))


with torch.no_grad():
  for data in testset:
    X,y = data
    output = net(X.view(-1,784))
    for idx, i in enumerate(output):
      if torch.argmax(i) == y[idx]:
        correct += 1
      total +=1
print(" Test accuracy: ", round(correct/total, 3))

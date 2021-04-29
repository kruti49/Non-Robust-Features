import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch as ch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from .models import *

learning_rate = 0.001
file_name = 'training_with_robust_dataset'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#create arguments for checkpoints
parser = argparse.ArgumentParser(description='PyTorch non-robust CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args, unknown = parser.parse_known_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#Data Augmentation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

#convert into Tensor dataset
class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

#Load dataset
data_path = "./datasets/zero_robust_CIFAR"
test_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
test_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
train_dataset = TensorDataset(test_data, test_labels, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

#Dataloader
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

#model
model = ResNet18()
model = model.to(device)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/zerorobust_ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

#optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=learning_rate)

def train(epoch):
    print('\n[ Train Epoch: %d ]' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent training batch:', str(batch_idx))

    print('\nTrain Accuarcy:', 100. *correct / total)
    print('\nTrain Loss:', train_loss / total)

def test(epoch):
    global best_acc
    print('\n[ Test Epoch: %d ]' % epoch)
    model.eval()
    test_correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(test_loader):
          inputs, targets = inputs.to(device), targets.to(device)
          total += targets.size(0)

          outputs = model(inputs)

          _, predicted = torch.max(outputs.data ,1)
          test_correct += predicted.eq(targets).sum().item()

          if batch_idx % 10 == 0:
            print('\nCurrent test batch:', str(batch_idx))
        
    print('\nTest Accuarcy:', 100. * test_correct / total)

    # Save checkpoint.
    acc = 100.*test_correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/zerorobustckpt.pth')
        best_acc = acc

#Decaying Learning rate
def lr_scheduler(optimizer, epoch):
        init_lr=0.001
        lr_decay_epoch=10
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1**(epoch / lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer
      
for epoch in range(start_epoch, 100):
    #lr_scheduler(optimizer,epoch)
    train(epoch)
    test(epoch)
    scheduler.step()

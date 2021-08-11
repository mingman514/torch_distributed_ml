"""
Synchronous SGD training on MNIST
Use distributed MPI backend
PyTorch distributed tutorial:
    http://pytorch.org/tutorials/intermediate/dist_tuto.html
This example make following updates upon the tutorial
1. Add params sync at beginning of each epoch
2. Allreduce gradients across ranks, not averaging
3. Sync the shuffled index during data partition
4. Remove torch.multiprocessing in __main__
"""
import os
import sys
import torch
import torch.utils.data                                                         
import torch.utils.data.distributed 
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from math import ceil
from random import Random
#from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel
from torchsummary import summary

from timeit import default_timer as timer

gbatch_size = 32
datapath = '/freeflow/shrd_datasets'
MASTER = 0 
TESTING = True

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # kernel size, stride, padding = 0 (default)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # input features, output features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_dataset():
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#            transforms.Normalize((0.1307,), (0.3081,))])

#    dataset = datasets.MNIST(
    dataset = datasets.CIFAR10(
        datapath,
        train=True,
        download=True,
        transform=transform)

    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)       
    train_set = torch.utils.data.DataLoader(                                    
        dataset, batch_size=bsz, shuffle=(train_sampler is None), sampler=train_sampler)
    return train_set, bsz

def init_model():
    # Select model
    model = Lenet5()
    if len(sys.argv) == 1 :
        model = Lenet5()
        model = DistributedDataParallel(model)
        print('###### Use default model [Lenet5] ######')
        return model

    _model = sys.argv[1]
    if _model == 'lenet5':
        model = Lenet5()
    elif _model == 'resnet18':
        model = models.resnet18()
    elif _model == 'resnet34':
        model = models.resnet34()
    elif _model == 'resnet50':
        model = models.resnet50()
    elif _model == 'resnet101':
        model = models.resnet101()
    elif _model == 'resnet152':
        model = models.resnet152()

    else:
        print('###### Incorrect model name ######')
        sys.exit()

    print('###### Use model {} ######'.format(_model))
    model = DistributedDataParallel(model)

    return model

def run(rank, size):
    print("RUN CODE STARTS")
    train_set, bsz = load_dataset()
    model = init_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    group = dist.new_group(list(range(size))) # all nodes join learning
    num_batches = ceil(len(train_set.dataset) / (float(bsz) * dist.get_world_size()))

    # Start training
    for epoch in range(1):
        epoch_loss = 0.0 

        for data, target in train_set:
            # slave compute the forward path 
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            model.zero_grad()
            loss.backward()

            # aggregates gradients 
            for param in model.parameters():
                dist.reduce(param.grad.data, MASTER, op=dist.ReduceOp.SUM, group=group) 
                # Average params in master node
                if rank == MASTER:
                    param.grad.data /= float(size)
                # Broadcast params to slave nodes
                dist.broadcast(param.grad.data, MASTER, group=group)
                optimizer.step()

            print('Rank #{} Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(rank,epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size()))

            if TESTING:
                break
                #sys.exit()

#    summary(model, (1, 28, 28))

if __name__ == "__main__":
    start_t = timer()
    dist.init_process_group(backend='mpi')
    mpi_ready_t = timer()

    size = dist.get_world_size()
    rank = dist.get_rank()
    print('size: {}  rank: {}'.format(size, rank))

    train_t = timer()
    run(rank, size)
    print('Program End')
    end_t = timer()
    
    if rank == MASTER:
        runtime = end_t - start_t
        mpitime = mpi_ready_t - start_t
        traintime = end_t - train_t
        print('MPI Init : {:.4f}s({:.2f}%)'.format(mpitime, mpitime/runtime*100))
        print('Learning : {:.4f}s({:.2f}%)'.format(traintime, traintime/runtime*100))
        print('  Total  : {:.4f}s({:.2f}%)'.format(runtime, runtime/runtime*100))

from scipy.linalg import orth
import numpy as np
import torch
import torch.utils.data
from torch.autograd.variable import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.nn.init
from torch.optim import Optimizer

from functools import reduce
import csv
import tables

import matplotlib.pyplot as plt
import seaborn


class VectorGenerator():
    
    A = None
    
    def __init__(self, m, n):
        np.random.seed(99)
        A = np.random.normal(size=(m, n))
        self.A = A
        
    def vec(self, k=1):
        A = self.A
        m, n = A.shape
        x = np.random.normal(size=(n, k))
        return np.dot(A, x).reshape((m, k))


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, alpha=0.01):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, alpha=alpha)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, model=None, inactive_subspace=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            alpha = group['alpha']

            if model is not None:
                flat_grad = model._flatten_grad().view(-1, 1)
                s = alpha * torch.norm(
                    inactive_subspace.transpose(0, 1) @ flat_grad
                )

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if inactive_subspace is not None:
                    assert model is not None
                    d_p.add_(s)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


class LeNet300_100(nn.Module):
    def __init__(self):
        super(LeNet300_100, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.output = nn.Linear(100, 10)
        self._numel_cache = None

    def forward(self, x):
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        x = self.output(x)
        return F.log_softmax(x)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(),
                                       self.parameters(),
                                       0)
        return self._numel_cache

    def _flatten_grad(self):
        views = []
        for p in self.parameters():
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _flatten_params(self):
        views = []
        for p in self.parameters():
            view = p.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _alter_params(self, update):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.mul_(0).add_(update[offset:offset + numel])
            offset += numel
        assert offset == self._numel()


if __name__ == '__main__':

    print('...Loading MNIST data')

    # load MNIST dataset
    batch_size = 5000
    cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    for data, target in train_loader:
        data, target = Variable(data, volatile=False), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        break
    data = data.view(-1, 28 * 28)  # no conv layers, so vec the images

    model = LeNet300_100()
    if cuda:
        model.cuda()

    def get_grad(a=-10.0, b=10.0):
        nParams = model._numel()

        newParams = torch.FloatTensor(nParams)
        torch.nn.init.uniform(newParams, a=a, b=b)
        if cuda:
            newParams = newParams.cuda()

        model._alter_params(newParams)
        model.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        return model._flatten_grad().cpu().numpy().reshape(model._numel(), 1)

    
    #m, n = 10, 4
    #vg = VectorGenerator(m, n)
    #print(vg.vec().shape)

    s_vals = []
    for k in range(1):
        # initialize
        k = 100
        X = orth(np.random.uniform(low=-2.0, high=2.0, size=(model._numel(), k)))

        print('...Computing inactive subspace')
        # sequentially generate vectors and update the subspace
        for i in range(100):
            if (i + 1) % 5 == 0:
                print('\t...Subspace update iteration {}'.format(i + 1))
            v = get_grad(a=-2.0, b=2.0)  # vg.vec()
            v = v / np.linalg.norm(v)
            X = X - np.dot(v, v.transpose().dot(X))
            X = orth(X)

        print('...Checking inactive subspace accuracy')
        # check that the subspace is good
        s = 0.
        for i in range(10):
            if (i + 1) % 10 == 0:
                print('\t...Subspace check iteration {}'.format(i + 1))
            v = get_grad(a=-2.0, b=2.0)  # vg.vec()
            s += np.linalg.norm( np.dot(X.transpose(), v) )**2

        s_vals.append(s)
        print('k: {:d}, s: {:6.4e}'.format(k, np.sqrt(s)))

    """
    seaborn.set_style('white')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.semilogy(np.arange(1, len(s_vals) + 1), s_vals,
                'k-o', linewidth=4.0, markersize=10.0)
    ax.set_xlabel('Inactive Subspace Estimate Dimension')
    ax.set_ylabel('Subspace Estimate Accuracy')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    """

    # train
    cuda = True
    log_interval = 10
    epochs = 10
    test_batch_size = 1000
    train_batch_size = 64

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        break

    model = LeNet300_100()
    init_params = model._flatten_params()
    if cuda:
        model.cuda()

    optimizer = SGD(model.parameters(), lr=0.01, alpha=1.0)

    # print(model._numel())
    # for batch_idx, (data, target) in enumerate(train_loader):
    #    data = Variable(data.cuda())
    #    output = model(data)

    losses1 = []
    losses2 = []
    def train1(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            losses1.append(loss.data[0])
            loss.backward()
            optimizer.step(model=model, inactive_subspace=torch.FloatTensor(X))
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

    def train2(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            losses2.append(loss.data[0])
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    for epoch in range(1, epochs + 1):
        train1(epoch)
        test(epoch)

    model = LeNet300_100()
    model._alter_params(init_params)
    optimizer = SGD(model.parameters(), lr=0.01)
    for epoch in range(1, epochs + 1):
        train2(epoch)
        test(epoch)

    seaborn.set_style('white')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.semilogy(np.arange(0, len(losses1)), losses1, linewidth=4.0)
    ax.semilogy(np.arange(0, len(losses2)), losses2, linewidth=4.0)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend(['Active Subspace SGD', 'Standard SGD'])
    plt.tight_layout()
    plt.show()

    trained_params = model._flatten_params().cpu().numpy()
    print('Trained params: \t max {} | min {}'.format(np.max(trained_params),
                                                      np.min(trained_params)))
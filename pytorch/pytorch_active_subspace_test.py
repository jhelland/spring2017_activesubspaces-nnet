
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

from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self._numel_cache = None
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 2)

    def forward(self, x):
        x = F.tanh(self.hidden(x))
        x = self.output(x)
        x = F.log_softmax(x)
        return x

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


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self._numel_cache = None
        self.output = nn.Linear(2, 2)

    def forward(self, x):
        x = self.output(x)
        x = F.log_softmax(x)
        return x

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
    print('...Generating data')
    N = 100

    t = np.linspace(0, 2 * np.pi, N)
    data = np.zeros((2, 2 * N))
    data[0, :] = np.concatenate([
        2 * np.cos(t) + np.random.normal(0, 0.15, t.size),
        4 * np.cos(t) + np.random.normal(0, 0.15, t.size)
    ])
    data[1, :] = np.concatenate([
        -2 * np.sin(t) + np.random.normal(0, 0.15, t.size),
        -4 * np.sin(t) + np.random.normal(0, 0.15, t.size)
    ])
    labels = np.concatenate([np.ones(N),
                             np.zeros(N)]).astype('int')
    data, target = Variable(torch.Tensor(data.T)), Variable(torch.LongTensor(labels))

    model = Net()
    nParams = model._numel()

    def get_grad(a=-10.0, b=10.0):
        #nParams = model._numel()

        newParams = torch.FloatTensor(nParams)
        torch.nn.init.uniform(newParams, a=a, b=b)

        model._alter_params(newParams)
        model.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        return model._flatten_grad().numpy().reshape(model._numel(), 1)


    # m, n = 10, 4
    # vg = VectorGenerator(m, n)
    # print(vg.vec().shape)

    s_vals = []
    for k in range(5, model._numel() + 1):
        # initialize
        # k = 1
        X = orth(np.random.normal(size=(model._numel(), k)))

        print('...Computing inactive subspace')
        # sequentially generate vectors and update the subspace
        for i in range(5000):
            #if (i + 1) % 20 == 0:
            #    print('\t...Subspace update iteration {}'.format(i + 1))
            v = get_grad(a=-2.0, b=2.0)  # vg.vec()
            v = v / np.linalg.norm(v)
            X = X - np.dot(v, v.transpose().dot(X))
            X = orth(X)

        print('...Checking inactive subspace accuracy')
        # check that the subspace is good
        s = 0.
        for i in range(100):
            #if (i + 1) % 20 == 0:
            #    print('\t...Subspace check iteration {}'.format(i + 1))
            v = get_grad(a=-2.0, b=2.0)  # vg.vec()
            s += np.linalg.norm(np.dot(X.transpose(), v)) ** 2

        print('k: {:d}, s: {:6.4e}'.format(k, np.sqrt(s)))
        s_vals.append(np.sqrt(s))

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
    n_bootstraps = 1
    eigs = torch.FloatTensor(n_bootstraps, nParams)
    for s in range(n_bootstraps):
        grad_matrix = torch.FloatTensor(nParams, 500)
        for i in range(500):
            # generate parameter sample
            new_params = torch.FloatTensor(nParams)
            torch.nn.init.uniform(new_params, a=-2.0, b=2.0)

            # compute gradient given new parameters
            model._alter_params(new_params)
            model.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            grad_matrix[:, i] = model._flatten_grad()

        U, eigs[s, :], _ = torch.svd(grad_matrix)
        eigs[s, :] = eigs[s, :]**2 / np.sqrt(nParams)

    
    stats = [
        torch.mean(eigs, dim=1),
        torch.min(eigs, dim=1),
        torch.max(eigs, dim=1)
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.semilogy(np.arange(0, len(eigs[0, :])), eigs[0, :].numpy(), 'k-o')

    plt.show()
    """

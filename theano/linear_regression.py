
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
from mpl_toolkits.mplot3d import Axes3D

# computation & data structures
import numpy as np
import theano
import theano.tensor as T

from copy import deepcopy

from utils import *


def generate_data(response_fn, n=1, n_obs=100):
    """
    Generate data according to a function with Gaussian noise in n dimensions.
    Currently adds noise to the inputs as well as outputs.
    :param response_fn:
    :param n: dimension of data
    :param n_obs: number of observations
    :return:
    """

    x = np.array([
        np.linspace(0, 10, n_obs) + np.random.normal(0, 1, n_obs)
        for _ in range(n)
    ])
    response = response_fn(x) + np.random.normal(0, 1, n_obs)
    data_matrix = np.vstack((x, np.ones(n_obs))).T
    return data_matrix, response


def plot_regression(x_data, response, beta_path, cost_fn, plot_type='2d', title=''):
    """
    Simple plotting function given output of linear regression.
    :param x_data: observations
    :param response: response data
    :param coeff: linear regression coefficients
    :param plot_type:
    :return:
    """

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.axis('off')

    beta = beta_path[-1]

    if plot_type is '2d':
        x_range = (min(x_data), max(x_data))
        x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0])//2)
        y = beta[0]*x + beta[1]

        ax = fig.add_subplot(121)
        ax.scatter(x_data, response, color='r')
        ax.plot(x, y, linewidth=2, color='k')
        ax.set_title(title)

        xx, yy = np.meshgrid(np.linspace(-5.0, 5.0, 50), np.linspace(-5.0, 5.0, 50))
        zz = np.array([cost_fn([xx_ij, yy_ij])
                       for xx_i, yy_i in zip(xx, yy)
                       for xx_ij, yy_ij in zip(xx_i, yy_i)]).reshape(xx.shape)
        beta_xy = np.array(beta_path)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(xx, yy, zz,
                        cmap=cm.jet)
        ax2.plot(beta_xy[:, 0], beta_xy[:, 1], [cost_fn(entry) for entry in beta_path],
                 'kX-', markersize=10, linewidth=3)
        ax2.view_init(27, -80)

    elif plot_type is '3d':
        xrange = (min(x_data[:, 0]), max(x_data[:, 0]))
        yrange = (min(x_data[:, 1]), max(x_data[:, 1]))
        xx, yy = np.meshgrid(np.arange(xrange[0], xrange[1], (xrange[1] - xrange[0]) // 2),
                             np.arange(yrange[0], yrange[1], (yrange[1] - yrange[0]) // 2))
        zz = beta[0] * xx + beta[1] * yy + beta[2]

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data[:, 0], x_data[:, 1], response, color='r')
        ax.plot_surface(xx, yy, zz,
                        rstride=1, cstride=1,
                        color='None', alpha=0.4)
        ax.set_title(title)
        ax.view_init(10, -50)
    else:
        print('ERROR: not implemented')
    plt.show()


def run_linear_regression(n_obs=100, learning_rate=0.001, num_iterations=5, method='GD', plot=False):

    # generate data
    data, response = generate_data(
        lambda x: 2.0 * sum(xi for xi in x) + 1.0,
        n=1,
        n_obs=n_obs
    )
    beta_inp = np.zeros(2)

    # define symbolic variables & functions
    print('...Building linear least squares model')
    X = T.matrix('X')
    Y = T.vector('Y')
    beta = T.vector('beta')
    cost = (1/n_obs) * T.nlinalg.norm(Y - T.dot(X, beta), ord=2)**2
    cost_grad = T.grad(cost=cost, wrt=[beta])
    cost_fn = theano.function(inputs=[beta],
                              outputs=cost,
                              givens={
                                  X: data,
                                  Y: response
                              })
    cost_grad_fn = theano.function(inputs=[beta],
                                   outputs=cost_grad,
                                   givens={
                                       X: data,
                                       Y: response
                                   })

    xx, yy = np.meshgrid(np.linspace(-20.0, 20.0, 100), np.linspace(-20.0, 20.0, 100))
    zz = np.array([cost_fn([xx_ij, yy_ij])
                   for xx_i, yy_i in zip(xx, yy)
                   for xx_ij, yy_ij in zip(xx_i, yy_i)]).reshape(xx.shape)
    xy = np.array([[xx_ij, yy_ij]
                   for xx_i, yy_i in zip(xx, yy)
                   for xx_ij, yy_ij in zip(xx_i, yy_i)])
    print(xy)

    n_samples = 100
    param_samples = np.random.uniform(low=-10.0, high=10.0, size=(2, n_samples))
    grad_samples = np.zeros(shape=(2, n_samples))
    for i in range(n_samples):
        grad = cost_grad_fn(param_samples[:, i])
        grad_samples[:, i] = grad / np.linalg.norm(grad)

    svd = np.linalg.svd(grad_samples, full_matrices=False, compute_uv=True)
    w = svd[0]
    eigs = (1 / n_samples) * svd[1]**2
    print(eigs)

    fig = plt.figure(figsize=plt.figaspect(2/3))
    plt.axis('off')
    fig.tight_layout(pad=0.5)

    ax2 = fig.add_subplot(234, projection='3d')
    ax2.plot_surface(xx, yy, zz,
                     cmap=cm.jet)
    ax2.view_init(27, -80)
    ax2.set_title('Cost Surface')

    ax = fig.add_subplot(235)
    ax.semilogy(np.arange(1, len(eigs)+1, 1), eigs, 'k-o', linewidth=2)
    ax.set_xlabel('Index')
    ax.set_ylabel(r'$\lambda_i$')
    ax.set_title('Eigenvalues')

    ax = fig.add_subplot(236)
    wx = np.dot(xy, w[:, 0])
    ax.scatter(wx, zz.flatten()[::-1], marker='.', color='black')
    ax.set_xlabel(r'$w_1^T\beta$')
    ax.set_ylabel(r'$Cost$')
    ax.set_title('Active Subspace')

    ax = fig.add_subplot(232)
    x = np.linspace(-1, 12, 100)
    ax.scatter(data[:, 0], response)
    ax.plot(x, 2.0*x + 1.0, color='k', linewidth=2)
    ax.set_xlabel('Observations')
    ax.set_ylabel('Response')
    ax.set_title(r'Regression Line $Y=X\beta$')

    plt.show()


#################################
# MAIN SCRIPT
#################################

if __name__ == "__main__":

    run_linear_regression(n_obs=10, learning_rate=0.002, num_iterations=5, method='SGD', plot=True)

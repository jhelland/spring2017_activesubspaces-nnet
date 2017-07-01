"""
General utility functions
"""

#########################
# LIBRARIES
#########################

from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import random
import operator
import functools
import sys
import seaborn

import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from nnet_classes import MLP


#########################
# FUNCTIONS
#########################

def load_data(dataset):
    """
    Loads the MNIST dataset or downloads if not found
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('...Loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def get_param_sample(params, dists, sample_size=1000):

    assert len(dists) == len(params) or len(dists) == 1

    samples = []
    for i in range(sample_size):
        sample = []
        for p in range(len(params)):
            sample.append(dists[p](params[p].shape))
        samples.append(sample)

    return samples


def get_param_sample_from_bounds(classifier, params, sample_size=1000, alpha=0.1):
    """
    Generate a random sample of parameters distributed uniform with bounds calculated from
    trained model parameter values.
    :param classifier: object containing trained parameter values. This is passed in so that
     we do not slow down computations by reading in saved parameter values from a file every
     time that we want to get a sample.
    :param params: parameters of the model with proper dimension formatting. We need them like
    this because the bias vectors need to be broadcastable by column which requires that vectors
    be stored as column vectors rather than the default numpy structure of shape=(m,)
    :param sample_size: how many samples we want to generate
    :param alpha: percentage interval bounds
    :return: random sample of parameters
    """

    # +/- alpha% distribution bounds
    bounds = bounds_from_params(classifier, alpha=alpha)

    # for dimensions
    # params = [param.get_value(borrow=True) for param in classifier.params]

    samples = []
    for i in range(sample_size):
        sample = []
        for p in range(len(params)):
            s = np.random.uniform(
                low=bounds[p][0],
                high=bounds[p][1],
                size=params[p].shape
            ).astype(theano.config.floatX)
            sample.append(s)
        samples.append(sample)

    return samples


def bounds_from_params(classifier, alpha):
    """
    Get properly formatted bounds for parameter distributions based on trained parameter values
    :param classifier: object containing trained parameters
    :param alpha: interval width for parameter sampling distribution centered around trained
     parameter values.
    :return:
    """

    # get trained parameter values
    params = [p.get_value(borrow=True) for p in classifier.params]

    # format vectors
    #for p in range(len(params)):
    #    if len(params[p].shape) == 1:
    #        params[p] = np.reshape(params[p], (params[p].shape[0], 1))
    #    else:
    #        params[p] = params[p].T

    # calculate bounds
    bounds = [(np.zeros(p.shape), np.zeros(p.shape)) for p in params]
    for i in range(len(params)):
        for j in range(params[i].shape[0]):
            bounds[i][0][j] = np.array([elem - abs(elem)*alpha
                                        for elem in np.nditer(params[i][j])]).astype(theano.config.floatX)
            bounds[i][1][j] = np.array([elem + abs(elem)*alpha
                                        for elem in np.nditer(params[i][j])]).astype(theano.config.floatX)

    return bounds


# helper function to vectorize the gradient function output
def vec_grad(grad_fn, x):
    grad = grad_fn(*x)
    return np.concatenate(
        [np.reshape(g, functools.reduce(operator.mul, g.shape, 1))
         for g in grad]
    )


def vec(A):
    return np.concatenate([a.T.reshape(a.size,) for a in A])

"""
def vec(matrices, axis=1):
    return np.concatenate([a.T.flatten() if axis else a.flatten()
                           for a in matrices])
"""

def compute_GTG(param_sample, grad_fn, sample_size):

    print('...Computing (G^T)(G) with sample size %i' % sample_size)

    # compute (G^T)(G) and normalize each entry by ||g_i||*||g_j|| under the 2-norm
    k_matrix = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
#        g1 = vec_grad(grad_fn, param_sample[i])
        g1 = vec(grad_fn(*param_sample[i]))
        g1_norm = np.linalg.norm(g1)
        k_matrix[i, i] = np.inner(g1, g1) / g1_norm ** 2
        for j in range(i + 1, sample_size):
#            g2 = vec_grad(grad_fn, param_sample[j])
            g2 = vec(grad_fn(*param_sample[i]))

            gtg = np.inner(g1, g2)
            denom = g1_norm * np.linalg.norm(g2)
            k_matrix[i, j] = gtg / denom
            k_matrix[j, i] = gtg / denom

    return k_matrix


# for models with lots of parameters
def get_GTG(classifier,
            params,
            dists,
            grad_fn,
            sample_size=1000,
            dist_type="default",
            alpha=0.1,
            bootstrap=True,
            boot_sample_size=10000):
    """
    Compute the prduct (G^T)(G) where G is the matrix of gradient vector samples evaluated
    at randomly sampled parameter distributions governed by some probability distribution.
    Note that using default dist_type with a small alpha results in a matrix that is close to rank 1.
    :param params: model parameters. Pretty much just used for dimensions
    :param dists: handles to parameter sampling distributions that can be evaluated.
    This parameter is only used if we want to explicitly define probability distributions
    rather than using uniform samples centered at pre-trained parameter values.
    :param grad_fn: function handle to an evaluatable Theano function of the gradient
    with respect to each parameter in the model.
    :param fname: path to file containing pre-trained parameter values. This is only used when
    we don't want to explicitly define sampling distributions for the parameters.
    :param sample_size: how large the (G^T)(G) matrix should be.
    :param dist_type: "custom" indicates that sampling distributions have been explicitly
    defined.
    :param alpha: the percentage bound (only used for when we don't explicitly define sampling
    distributions).
    :param bootstrap: boolean indicating whether or not to compute bootstrap samples of the eigenvalues
    :return: (G^T)(G)
    """

    # compute full sample of parameters
    if dist_type is 'custom':
        param_sample = get_param_sample(params=params,
                                        dists=dists,
                                        sample_size=sample_size)
    else:
        param_sample = get_param_sample_from_bounds(classifier=classifier,
                                                    params=params,
                                                    sample_size=sample_size,
                                                    alpha=alpha)

    if bootstrap:
        return (
            compute_GTG(param_sample,
                        grad_fn,
                        sample_size),
            get_eig_bootstrap(param_sample,
                              grad_fn,
                              sample_size=sample_size,
                              boot_sample_size=boot_sample_size,
                              print_iteration=True)
        )
    else:
        return compute_GTG(param_sample,
                           grad_fn,
                           sample_size)


def get_eig_bootstrap(param_sample, grad_fn, sample_size, boot_sample_size, print_iteration=True):

    # Note, np.linalg.eigvalsh() is a function the exploits the properties of symmetric
    # or Hermitian matrices when computing eigenvalues. This is useful here because
    # (G^T)(G) is theoretically symmetric, but numerical errors can result in a matrix
    # with complex eigenvalues that have an imaginary component on the order of
    # floating point precision.

    print('......Computing eigenvalue bootstrap standard error intervals')
    boot_sample = np.zeros((boot_sample_size, sample_size))
    for i in range(boot_sample_size):
        indices = np.random.randint(0, sample_size, sample_size)
        boot_param_sample = [param_sample[i] for i in indices]
        boot_sample[i, :] = np.linalg.eigvalsh(
            compute_GTG(boot_param_sample, grad_fn, sample_size=sample_size)
        )

        if print_iteration:
            print('\t\tBootstrap iteration %i/%i' % (i+1, boot_sample_size))

    return boot_sample


def plot_eigs_from_file(fin_name, bootstrap=False, file_format='show', titles=['']):
    """
    TODO: fix broken bootstrap confidence intervals

    Plot the sorted eigenvalues from a stored matrix of type (G^T)(G).
    :param fname: path to stored matrix
    """

    print()
    print('...Loading matrices from %s' % fin_name)
    gtg_matrices = np.load(fin_name)
    if bootstrap:
        eig_intervals = []
        gtg_bootstrap_samples = np.load(fin_name[:-4] + '_bootstrap.npz')
        for _, gtg_boots in gtg_bootstrap_samples.items():
            for gtg_boot in gtg_boots:
                gtg_boot = np.sort(gtg_boot, axis=0)[:, ::-1]
                for i in range(len(gtg_boot)):
                    for j in range(len(gtg_boot[i])):
                        if gtg_boot[i, j] < 10**-16:
                            gtg_boot[i, j] = 10**-16
                eig_intervals.append(np.std(gtg_boot, axis=0))

        eig_intervals = np.array(eig_intervals)
        #print(eig_intervals)


    # either show or save each eigenvalue plot
    count = 0
    for _, gtg_matrices in gtg_matrices.items():

        for gtg_matrix in gtg_matrices:

            # compute eigenvalues
            print('...Computing eigenvalues %s' % (' bootstrap standard error intervals' if bootstrap else ''))
            eigenvals = np.sort(np.linalg.eigvalsh(gtg_matrix))[::-1]
            for i in range(len(eigenvals)):
                if eigenvals[i] < 10**-16:
                    eigenvals[i] = 10**-16

            print('...Plotting')
            plt.figure(figsize=(10, 8))
            plt.semilogy(range(1, len(eigenvals)+1), eigenvals,
                         'ko-', markersize=4, linewidth=1)
            if bootstrap:
                lower = eigenvals
                upper = eigenvals + 2*eig_intervals[count]
                for i in range(len(eigenvals)):
                    if eigenvals[i]:
                        lower[i] = eigenvals[i] - 2*eig_intervals[count][i]

                plt.fill_between(range(1, len(eigenvals)+1),
                                 lower, upper,
                                 facecolor="0.7", interpolate=True)
            plt.grid(True)
            plt.ylim([10**-16, 10**2])
            plt.xlabel('Index')
            plt.ylabel('Eigenvalues')
            plt.title(titles[count])

            if file_format is 'show':
                print('......Showing plot')
                plt.show()
            else:
                print('......Saving plot as ./figures/eigvals_alpha' + str(count) + file_format)
                plt.savefig('./figures/eigvals_alpha' + str(count) + file_format,
                            bbox_inches='tight')

            plt.clf()
            count += 1


def save_model_parameters_theano(model, outfile):
    """
    Save model parameters to .npz file. Currently only set up to work with MLP class model.
    :param model: object containing model parameters
    :param outfile: string of file output path
    """
    params = [param.get_value() for param in model.params]
    np.savez(outfile,
             *params)
    print("...Saved model parameters to %s." % outfile)


def load_model_parameters_theano(path, modelClass):
    """
    Load the model parameters from a .npz file.
    :param path: string of path to .npz file
    :param modelClass: handle to desired neural network class
    :return: built model
    """

    npzfile = np.load(path)

    W = []
    b = []
    count = 0
    for key in sorted(npzfile):
        if count % 2 is 0:
            W.append(npzfile[key])
        else:
            b.append(npzfile[key])
        count += 1

    layer_sizes = [param.shape[0] for param in W]
    layer_sizes.append(b[-1].shape[0])

    model = modelClass(
        rng=np.random.RandomState(1234),
        input=T.matrix('x'),
        layer_sizes=layer_sizes,
        W=W,
        b=b
    )

    return model

"""
Script to search for active subspaces in the MNIST MLP cost function
"""

#########################
# LIBRARIES
#########################

import timeit
import matplotlib.pyplot as plt
import csv
import os
import sys
import seaborn

import numpy as np
import theano
import theano.tensor as T

from utils import *
from nnet_classes import *
from gradient_descent_methods import *
from theano.compile.nanguardmode import NanGuardMode   # for returning errors when Theano computes a nan value (debug)


#########################
# METHODS
#########################

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500,
             update_rule='standard'):
    """
        Demonstrate stochastic gradient descent optimization for a multilayer
        perceptron
        This is demonstrated on MNIST.
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient
        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)
        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        :type dataset: string
        :param dataset: the path of the MNIST dataset file from
                     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
        : type update_rule: string
        : param update_rule: the method of updating the weights, either RMS, momentum, or nesterov
       """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as vectorized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        layer_sizes=[28*28, 300, 100, 10],
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
    #    + L1_reg * classifier.L1
    #    + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    # if update_rule != 'nesterov':
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # init epoch here so it can be used to smoothly scale up momentum
    epoch = 0

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    # if update_rule == 'standard':
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]
    # elif update_rule == 'RMS':
    #    updates = RMSprop(classifier.params, gparams, classifier.accs, lr=learning_rate)
    # elif update_rule == 'momentum':
    #    updates = classical_momentum(classifier.params, gparams, classifier.accs, epoch, n_epochs, lr=learning_rate)
    # elif update_rule == 'nesterov':
    #    updates = nesterov_momentum(classifier.params, classifier.accs, epoch, n_epochs, cost, lr=learning_rate)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    validation_errors = []
    while (epoch < n_epochs):  # and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                validation_errors.append(this_validation_loss * 100)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # save the best model
                    save_model_parameters_theano(classifier, './best_model_mlp.npz')

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('\t epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                # break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    with open('optimization_%s%f.csv' % (update_rule, learning_rate), 'w') as csvfile:
        fieldnames = ['error_validation_set',
                      'val_freq',
                      'minibatch/epoch',
                      'batch_size',
                      'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict([('error_validation_set', validation_errors),
                              ('val_freq', validation_frequency),
                              ('minibatch/epoch', n_train_batches),
                              ('batch_size', batch_size),
                              ('learning_rate', learning_rate)]))


def predict(first_ten=True, dataset='mnist.pkl.gz'):
    """
    Computes the prediction of the MLP for some input data. Also prints what the actual values should be for reference.
    """

    # load the saved model
    classifier = load_model_parameters_theano('./best_model_mlp.npz', MLP)

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)

    # We can test it on some examples from test test
    #A = load_data("test.csv")
    #print("loading test data....")
    #A.read_test_file()

    #test_set_x = A.inputs
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    if first_ten:
        predicted_values = predict_model(test_set_x[:10])
    else:
        predicted_values = predict_model(test_set_x[-10:])

    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print("Labels")
    print(test_set_y.eval()[-10:])

    """
    # print maximum and minimum values for each parameter
    for p in classifier.params:
        p_val = p.get_value(borrow=True)
        print([np.min(p_val), np.max(p_val)])
    """

    return predicted_values


def get_mnist_gtg(dataset='mnist.pkl.gz',
                  fname_out='gtg_matrix',
                  sample_size=1000,
                  bootstrap=False,
                  boot_sample_size=100,
                  alphas=[0.1]):
    """
    TODO: - give this function a better name.
    - more parameter sampling distribution options
    - should gradients be evaluated for minibatches?

    Calculates the (G^T)(G) matrix for characterizing the active subspace. G is the matrix of gradient samples.
     To compute this matrix, we must draw samples from the sampling distributions of the parameters.
     Note: this MNIST classifier has 397,510 parameters that must be trained.
    :param dataset: MNIST image data
    :param sample_size: Number of gradient samples. Corresponds to the number of eigenvalues we want to observe.
    This will be the dimension of (G^T)(G).
    :return: (G^T)(G)
    """

    # load classifier
    classifier = load_model_parameters_theano('./best_model_mlp.npz', MLP)


    # load data
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value(borrow=True).T[:, :50]
    test_set_y = test_set_y.eval()[:50]

    del datasets

    print('...Initializing variables')

    # define symbolic variables
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(None)
    classifier = MLP(
        rng=rng,
        input=x,
        layer_sizes=[28 * 28, 500, 10]
    )

    params = classifier.params_sym
    cost = classifier.negative_log_likelihood_sym(y)

    grad = T.grad(cost, wrt=params)
    grad_fn = theano.function(inputs=params,
                              outputs=grad,
                              givens={
                                  x: test_set_x.T,
                                  y: test_set_y
                              },
                              mode=NanGuardMode(nan_is_error=True))

    print('...Initializing input parameters')

    # dimensions of network layers
    params_inp = [param.eval().astype(theano.config.floatX) for param in classifier.params]

    # define parameter sampling distributions
    dists = [lambda n: np.random.uniform(low=-5.0, high=5.0, size=n).astype(theano.config.floatX)
             for _ in classifier.params]

    # get gradient sample (note: there are 397,510 parameters)
    print('...Computing gradient sample')

    GTG = []
    for i in range(len(alphas)):

        print('alpha_%i = %f' % (i, alphas[i]))

        start_time = timeit.default_timer()
        GTG.append(get_GTG(classifier,
                           params=params_inp,
                           dists=dists,
                           grad_fn=grad_fn,
                           sample_size=sample_size,
                           dist_type='custom',
                           alpha=alphas[i],
                           bootstrap=bootstrap,
                           boot_sample_size=boot_sample_size))
        end_time = timeit.default_timer()
        print('......(G^T)(G) computed in %.2fm' % ((end_time - start_time) / 60.0))

        # save (G^T)(G) to disk
        """
        print('...Saving (G^T)(G) matrix to %s.npy' % fname_out)
        if bootstrap:
            GTG_matrix, GTG_eig_boot = GTG
            np.save(fname_out, GTG_matrix)

            print('...Saving eigenvalue bootstrap samples to %s.npy' % (fname_out + '_bootstrap'))
            np.save(fname_out + '_bootstrap', GTG_eig_boot)
        else:
            np.save(fname_out, GTG)
        """
    # assume bootstrap == False for now
    np.savez(fname_out, GTG)


def cluster1_test(data_fname=None, learning_rate=0.01, n_iterations=1000, showplots=False, data_type='gaussian'):

    if data_fname is None:
        print('...Generating cluster data')
        N = 100
        labels = np.concatenate([np.ones(N),
                                 np.zeros(N)])
        if data_type == 'gaussian':
            labels = np.random.random_integers(0, 1, 20)
            means = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
            temp_matrix = np.array([np.random.random_sample((3, 3)),
                                    np.random.random_sample((3, 3))])
            covariances = np.array([np.dot(temp_matrix[0].T, temp_matrix[0]),
                                    np.dot(temp_matrix[1].T, temp_matrix[1])])
            data = np.vstack([np.random.multivariate_normal(means[i], covariances[i])
                             for i in labels]).T
            data = data[0:2, :]
        elif data_type == 'spiral':
            t = np.linspace(0, 6, N)
            data = np.zeros((2, 2*N))
            data[0, :] = np.concatenate([
                -np.cos(t)*t + np.random.normal(0, 0.1, t.size),
                np.cos(t)*t + np.random.normal(0, 0.1, t.size)
            ])
            data[1, :] = np.concatenate([
                np.sin(t)*t + np.random.normal(0, 0.1, t.size),
                -np.sin(t)*t + np.random.normal(0, 0.1, t.size)
            ])
        elif data_type == 'circle':
            t = np.linspace(0, 2*np.pi, N)
            data = np.zeros((2, 2 * N))
            data[0, :] = np.concatenate([
                np.cos(t) + np.random.normal(0, 0.1, t.size),
                4*np.cos(t) + np.random.normal(0, 0.1, t.size)
            ])
            data[1, :] = np.concatenate([
                -np.sin(t) + np.random.normal(0, 0.1, t.size),
                -4*np.sin(t) + np.random.normal(0, 0.1, t.size)
            ])

        data = data.astype(theano.config.floatX)
        labels = labels.astype('int32')

        print('...Saving cluster data')
        np.savez('./cluster1_cluster_data', data, labels)
    else:
        print('...Loading data')
        with np.load(data_fname + '.npz') as data_file:
            data = data_file['arr_0'].astype(theano.config.floatX)
            labels = data_file['arr_1'].astype('int32')
        N = data[0, :].size

    print('...Building model')
    # initialize symbolic parameters
    x = T.matrix('x_input')
    y = T.ivector('y_label')

    rng = np.random.RandomState(None)

    classifier = MLP(
        rng=rng,
        input=x,
        layer_sizes=[2, 6, 2, 2]
    )

    cost = (
        classifier.negative_log_likelihood(y)
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[classifier.input],
        outputs=cost,
        updates=updates,
        givens={
            y: labels
        }
    )

    classify = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred
    )

    output_layer = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.output
    )

    print('...Training model for %i iterations' % n_iterations)
    accuracies = np.zeros(n_iterations)
    accuracies2 = np.zeros(n_iterations)
    for iteration in range(n_iterations):
        train_model(data.T)
        current_output = classify(data.T)
        accuracies[iteration] = np.mean((current_output > 0.5) == labels[:])

    if showplots:
        xx, yy = np.mgrid[-8:8:.01, -8:8:.01]
        prob_grid = np.c_[xx.ravel(), yy.ravel()]
        probs = classify(prob_grid).reshape(xx.shape)

        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        seaborn.set(style='white')

        axs = fig.add_subplot(221)
        axs.plot(np.linspace(1, len(accuracies), len(accuracies)), accuracies, 'k-')
        axs.set_xlabel('Iteration')
        axs.set_ylabel('Accuracy')
        axs.set_title('Training Accuracy')

        axs = fig.add_subplot(222)
        axs.scatter(data[0, :], data[1, :],
                    c=labels, cmap='RdBu', edgecolor='white')
        axs.set_title('Unclassified Data')
        axs.set_xlabel(r'$x_1$')
        axs.set_ylabel(r'$x_2$')

        axs = fig.add_subplot(223)
        axs.contourf(xx, yy, probs, 25, cmap='RdBu',
                    vmin=0, vmax=1, alpha=0.5)
        axs.scatter(data[0, :], data[1, :],
                    c=current_output, cmap='RdBu', edgecolor='white',
                    vmin=0, vmax=1)
        axs.set_title('Model Classification')
        axs.set_xlabel(r'$x_1$')
        axs.set_ylabel(r'$x_2$')

        output = output_layer(data.T)
        axs = fig.add_subplot(224)
        axs.scatter(output[:, 0], output[:, 1],
                    c=labels[:], cmap='RdBu',
                    edgecolor='white')
        axs.set_title('Transformed Data')
        axs.set_xlabel(r'$M(X)_1$')
        axs.set_ylabel(r'$M(X)_2$')

        #plt.savefig('../cluster1_classified_data.png', bbox_inches='tight')
        plt.show()

    print('...Saving model with %i parameters and cluster data' % classifier.n_params)
    save_model_parameters_theano(classifier, './cluster1_model.npz')


def cluster1_gtg(fname_data='./cluster1_cluster_data.npz',
                 fname_model='./cluster1_model.npz',
                 fname_out='./cluster1_gtg_matrix',
                 alphas=[0.1],
                 sample_size=100,
                 bootstrap=True,
                 boot_sample_size=10000):

    print('...Loading data')
    with np.load(fname_data) as data_file:
        data = data_file['arr_0'].astype(theano.config.floatX)
        labels = data_file['arr_1'].astype('int32')

    print('...Loading model parameters')
    classifier = load_model_parameters_theano(fname_model, MLP)

    print('...Initializing gradient function')
    y = T.ivector('y_input')

    grad = T.grad(cost=classifier.negative_log_likelihood_sym(y), wrt=classifier.params_sym)
    grad_fn = theano.function(inputs=classifier.params_sym, #params_symb,
                              outputs=grad,
                              givens={
                                  classifier.input: data.T,
                                  y: labels
                              })

    print('...Initializing ')
    # initialize parameters with uniform random values
    shapes = [param.get_value().shape for param in classifier.params]
    params_inp = [np.random.uniform(0, 1, shape).astype(theano.config.floatX) for shape in shapes]

    # define parameter sampling distributions
    dists = [lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             #lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             #lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX)]

    # get gradient sample (note: there are 397,510 parameters)
    print('...Computing gradient sample')

    GTG = []
    GTG_boot = []
    for i in range(len(alphas)):
        print('alpha_%i = %f' % (i, alphas[i]))

        start_time = timeit.default_timer()
        GTG_temp = (get_GTG(classifier=classifier,
                            params=params_inp,
                            dists=dists,
                            grad_fn=grad_fn,
                            sample_size=sample_size,
                            dist_type='custom',
                            alpha=alphas[i],
                            bootstrap=bootstrap,
                            boot_sample_size=boot_sample_size))
        end_time = timeit.default_timer()
        print('......(G^T)(G) computed in %.2fm' % ((end_time - start_time) / 60.0))

        # save (G^T)(G) to disk
        print('...Saving (G^T)(G) matrix to %s.npy' % fname_out)
        if bootstrap:
            GTG_matrix, GTG_eig_boot = GTG_temp
            GTG.append(GTG_matrix)
            GTG_boot.append(GTG_eig_boot)
        else:
            GTG.append(GTG_temp)
    # assume bootstrap == False for now
    np.savez(fname_out, GTG)
    if bootstrap:
        np.savez(fname_out + '_bootstrap', GTG_boot)


def gradient_test(fname_out='gtest_gtg_matrix', boot_sample_size=100):

    # generate two gaussian clusters in R^2
    np.random.seed(None)
    N = 1000  # number of data points

    y = np.random.random_integers(0, 1, N)
    means = np.array([[-1, 1], [-1, 1]])
    covariances = np.random.random_sample((2, 2)) + 1
    X = np.vstack([np.random.randn(N) * covariances[0, y] + means[0, y],
                   np.random.randn(N) * covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
    y = y.astype(theano.config.floatX)

    W_inp = [np.random.rand(4, 2).astype(theano.config.floatX)]
    W_inp.append(np.random.rand(1, 4).astype(theano.config.floatX))
    b_inp = [np.random.rand(4, 1).astype(theano.config.floatX)]
    b_inp.append(np.random.rand(1, 1).astype(theano.config.floatX))

    # initial parameter values
    params = [W_inp[0],
              b_inp[0],
              W_inp[1],
              b_inp[1]]

    # parameter sampling distributions
    dists = [lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=0.1, high=1.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=-10.0, high=10.0, size=n).astype(theano.config.floatX),
             lambda n: np.random.uniform(low=0.1, high=1.0, size=n).astype(theano.config.floatX)]

    # symbolic parameters for theano function
    # note: broadcastable allows b1/b2 to be added to each network output column
    W1 = T.matrix('W1')
    b1 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b1')
    W2 = T.matrix('W2')
    b2 = T.TensorType(dtype=theano.config.floatX,
                      broadcastable=(False, True))('b2')
    x = T.matrix('x')
    y_var = T.vector('y')

    # symbolic cost function
    cost = T.sum((T.nnet.sigmoid(
        T.dot(W2, T.nnet.sigmoid(T.dot(W1, x) + b1))
        + b2) - y_var) ** 2)

    # theano symbolic gradient computation
    grad = T.grad(cost=cost, wrt=[W1, b1, W2, b2])
    grad_fn = theano.function(inputs=[W1, b1, W2, b2],
                              outputs=grad,
                              givens={
                                  x: X,
                                  y_var: y
                              },
                              mode=NanGuardMode(nan_is_error=True))

    # compute the formatted gradient matrix using the specified probability distributions
    start_time = timeit.default_timer()
    GTG = get_GTG(params=params,
                  dists=dists,
                  grad_fn=grad_fn,
                  sample_size=10,
                  dist_type="default",
                  bootstrap=True,
                  boot_sample_size=boot_sample_size)
    end_time = timeit.default_timer()
    print('......(G^T)(G) computed in %.2fm' % ((end_time - start_time) / 60.0))

    # save matrices
    GTG_matrix, GTG_eig_boot = GTG
    np.save(fname_out, GTG_matrix)

    print('...Saving eigenvalue bootstrap samples to %s.npy' % (fname_out + '_bootstrap'))
    np.save(fname_out + '_bootstrap', GTG_eig_boot)

    plot_eigs_from_file(fin_name=fname_out + '.npy', bootstrap=True)



############################
# MAIN SCRIPT
############################

def make_gif(fin_dir, fin_name, fout_name, duration=5):
    import imageio

    images = []
    for i in range(len(alphas)):
        images.append(imageio.imread((fin_dir + fin_name + '%i.png') % i))
    kargs = {'duration': duration}
    imageio.mimsave(fout_name, images, 'GIF', **kargs)

#### quadratic cluster data
"""
fname = './cluster1_cluster_data'
cluster1_test(data_fname=None,
              showplots=True,
              learning_rate=0.1,
              n_iterations=20000,
              data_type='gaussian')
#alphas = np.logspace(-5, 1, 5).astype(theano.config.floatX)
cluster1_gtg(alphas=[10], sample_size=150, bootstrap=False, boot_sample_size=10)
#titles = [r'$\alpha = $%f' % alpha for alpha in alphas]
titles = [r'$\mathcal{U}(-10, 10)$']
plot_eigs_from_file(fin_name='./cluster1_gtg_matrix.npz', bootstrap=False, file_format='show', titles=titles)
# make_gif(fin_dir='./figures/', fin_name='eigvals_alpha', fout_name='./cluster1_eigvals.gif', duration=5)
"""


#### MNIST single hidden layer model
# train
# test_mlp(n_epochs=1000, update_rule='standard', learning_rate=0.01)

# predict
# x = predict(first_ten=False)

# compute (G^T)(G) where G is the matrix of gradient samples
# alphas = np.arange(0.00, 0.10, 0.05).astype(theano.config.floatX)

get_mnist_gtg(dataset='mnist.pkl.gz',
              fname_out='gtg_matrices',
              sample_size=100,
              bootstrap=False,
              boot_sample_size=50,
              alphas=[0])


# load a saved matrix and plot the eigenvalues with bootstrap standard error intervals
# plot_eigs_from_file(fin_name='gtg_matrix2.npy', bootstrap=False, format='.png')
# titles = [r'$\alpha = %f$' % alpha for alpha in alphas]
titles = [r'$\theta \sim \mathcal{U}(-10, 10)$']
plot_eigs_from_file(fin_name='gtg_matrices.npz',
                    bootstrap=False,
                    file_format='show',
                    titles=titles)


# just testing bootstrap intervals on a small example
# gradient_test(boot_sample_size=100)

# make gif
# TODO: fixed axis, label indicating alpha values
#import imageio
#images = []
#for i in range(len(alphas)):
#    images.append(imageio.imread('./figures/eigvals_alpha%i.png' % i))
#kargs = {'duration': 0.5}
#imageio.mimsave('./eigvals2.gif', images, 'GIF', **kargs)


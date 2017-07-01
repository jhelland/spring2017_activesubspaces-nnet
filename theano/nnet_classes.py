"""
Theano integrated neural network layer classes for defining multi-layer perceptron models.
"""

#########################
# LIBRARIES
#########################

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T


#########################
# CLASSES
#########################


class Layer(object):
     def __init__(self, W_init, b_init, activation):
         '''
         A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
         :parameters:
         - W_init : np.ndarray, shape=(n_output, n_input)
         Values to initialize the weight matrix to.
         - b_init : np.ndarray, shape=(n_output,)
         Values to initialize the bias vector
         - activation : theano.tensor.elemwise.Elemwise
         Activation function for layer output
         '''
         # Retrieve the input and output dimensionality based on W's initialization
         n_output, n_input = W_init.shape
         # Make sure b is n_output in size
         assert b_init.shape == (n_output,)
         # All parameters should be shared variables.
         # They're used in this class to compute the layer output,
         # but are updated elsewhere when optimizing the network parameters.
         # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
         self.W = theano.shared(value=W_init.astype(theano.config.floatX),
         # The name parameter is solely for printing purporses
                                name='W',
        # Setting borrow=True allows Theano to use user memory for this object.
        # It can make code slightly faster by avoiding a deep copy on construction.
        # For more details, see
        # http://deeplearning.net/software/theano/tutorial/aliasing.html
                                borrow=True)
         # We can force our bias vector b to be a column vector using numpy's reshape method.
         # When b is a column vector, we can pass a matrix-shaped input to the layer
         # and get a matrix-shaped output, thanks to broadcasting (described below)
         self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX),
                                name='b',
                                borrow=True,
        # Theano allows for broadcasting, similar to numpy.
        # However, you need to explicitly denote which axes can be broadcasted.
        # By setting broadcastable=(False, True), we are denoting that b
        # can be broadcast (copied) along its second dimension in order to be
        # added to another variable. For more information, see
        # http://deeplearning.net/software/theano/library/tensor/basic.html
                                broadcastable=(False, True))
         self.activation = activation
         # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
         self.params = [self.W, self.b]

     def output(self, x):
         '''
         Compute this layer's output given an input

         :parameters:
         - x : theano.tensor.var.TensorVariable
         Theano symbolic variable for layer input
         :returns:
         - output : theano.tensor.var.TensorVariable
         Mixed, biased, and activated x
         '''
         # Compute linear mix
         lin_output = T.dot(self.W, x) + self.b
         # Output is just linear mix if no activation function
         # Otherwise, apply the activation function
         return (lin_output if self.activation is None else self.activation(lin_output))


class MLP2(object):
     def __init__(self, W_init, b_init, activations):
         '''
         Multi-layer perceptron class, computes the composition of a sequence of Layers
         :parameters:
         - W_init : list of np.ndarray, len=N
         Values to initialize the weight matrix in each layer to.
         The layer sizes will be inferred from the shape of each matrix in W_init
         - b_init : list of np.ndarray, len=N
         Values to initialize the bias vector in each layer to
         - activations : list of theano.tensor.elemwise.Elemwise, len=N
         Activation function for layer output for each layer
         '''
         # Make sure the input lists are all of the same length
         assert len(W_init) == len(b_init) == len(activations)

         # Initialize lists of layers
         self.layers = []
         # Construct the layers
         for W, b, activation in zip(W_init, b_init, activations):
             self.layers.append(Layer(W, b, activation))

         # Combine parameters from all layers
         self.params = []
         for layer in self.layers:
             self.params += layer.params

     def output(self, x):
         '''
         Compute the MLP's output given an input

         :parameters:
         - x : theano.tensor.var.TensorVariable
         Theano symbolic variable for network input
         :returns:
         - output : theano.tensor.var.TensorVariable
         x passed through the MLP
         '''
         # Recursively compute output
         for layer in self.layers:
             x = layer.output(x)
         return x

     def squared_error(self, x, y):
         '''
         Compute the squared euclidean error of the network output against the "true" output y

         :parameters:
         - x : theano.tensor.var.TensorVariable
         Theano symbolic variable for network input
         - y : theano.tensor.var.TensorVariable
         Theano symbolic variable for desired network output
         :returns:
         - error : theano.tensor.var.TensorVariable
         The squared Euclidian distance between the network output and y
         '''
         return T.sum((self.output(x) - y)**2)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, input_sym, n_in=None, n_out=None, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # the borrow parameter is for the sake of Theano memory optimization
        # theano.config.floatX is a value set in the ~/.theanorc file: probably float32
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W_output',
                borrow=True
            )
        else:
            self.W = theano.shared(
                value=W.astype(theano.config.floatX),
                name='W_output',
                borrow=True
            )

        # initialize the biases b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b_output',
                borrow=True
            )
        else:
            self.b = theano.shared(
                value=b.astype(theano.config.floatX),
                name='b_output',
                borrow=True
            )

        self.W_sym = T.matrix('W_out_sym')
        self.b_sym = T.vector('b_out_sym')#T.TensorType(dtype=theano.config.floatX,
                     #             broadcastable=(False, True))('b_out_sym')

        def softmax(x):
            """
            Numerically stable version of the log(softmax(x)) function. We need this because
            Theano's version has numerical stability issues that can cause NaN values for
            gradient evaluations.
            :param x: input for activation function. Probably a matrix.
            :return: log(softmax(x)) - log probability outputs for each input
            """
            e_x = T.exp(x - x.max(axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.output = T.dot(input, self.W) + self.b

        self.p_y_given_x = softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x_sym = softmax(T.dot(input_sym, self.W_sym) + self.b_sym)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        self.params_sym = [self.W_sym, self.b_sym]

        # keep track of model input
        self.input = input
        self.input_sym = input_sym

    def negative_log_likelihood_sym(self, y):
        return -T.mean(self.p_y_given_x_sym[T.arange(y.shape[0]), y])

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(self.p_y_given_x[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):

    def __init__(self, rng, input, input_sym, n_in=None, n_out=None, W=None, b=None,
                 activation=T.tanh):

        self.input = input
        self.input_sym = input_sym

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W_hidden', borrow=True)
        else:
            self.W = theano.shared(value=W.astype(theano.config.floatX),
                                   name='W_hidden',
                                   borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_hidden', borrow=True)
        else:
            self.b = theano.shared(value=b.astype(theano.config.floatX),
                                   name='b_hidden',
                                   borrow=True)

        self.W_sym = T.matrix('W_hidden_sym')
        self.b_sym = T.vector('b_hidden_sym')#T.TensorType(dtype=theano.config.floatX,
                     #             broadcastable=(False, True))('b_hidden_sym')

        lin_output = T.dot(input, self.W) + self.b
        lin_output_sym = T.dot(input_sym, self.W_sym) + self.b_sym
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.output_sym = (
            lin_output_sym if activation is None
            else activation(lin_output_sym)
        )

        # parameters
        self.params = [self.W, self.b]
        self.params_sym = [self.W_sym, self.b_sym]


class MLP(object):

    # def __init__(self, rng, input, n_in, n_hidden, n_out, W=None, b=None, regularize=True):
    def __init__(self, rng, input, layer_sizes, W=None, b=None, regularize=True):
        """
        :param rng:
        :param input: MNIST digits minibatch
        :param n_in: number of input nodes (28*28)
        :param n_hidden: number of hidden nodes
        :param n_out: number of output nodes (10)
        :param W: two element list containing W_hidden and W_output
        :param b: two element list containing b_hidden and b_output
        :param regularize: if true, then use L1 and L2^2 regularization terms when training
        """

        self.hiddenLayer = []

        # we use a tanh activation layer as the hidden layer
        temp_input = [input]
        temp_input_sym = [input]
        for i in range(len(layer_sizes)-2):
            if W is None:
                self.hiddenLayer.append(HiddenLayer(
                    rng=rng,
                    input=temp_input[i],
                    input_sym=temp_input_sym[i],
                    n_in=layer_sizes[i],
                    n_out=layer_sizes[i+1],
                    activation=T.tanh
                ))
            else:
                self.hiddenLayer.append(HiddenLayer(
                    rng=rng,
                    input=temp_input[i],
                    input_sym=temp_input_sym[i],
                    activation=T.tanh,
                    W=W[i],
                    b=b[i]
                ))
            temp_input.append(self.hiddenLayer[i].output)
            temp_input_sym.append(self.hiddenLayer[i].output_sym)

        # we use a logistic regression layer as the final layer
        if W is None:
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer[-1].output,
                input_sym=self.hiddenLayer[-1].output_sym,
                n_in=layer_sizes[-2],
                n_out=layer_sizes[-1]
            )
        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer[-1].output,
                input_sym=self.hiddenLayer[-1].output_sym,
                W=W[-1],
                b=b[-1]
            )

        # we use L1 and L2 regularization
        """
        if regularize:
            self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.logRegressionLayer.W).sum()
            )

            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.logRegressionLayer.W ** 2).sum()
            )
        else:
            self.L1 = 0
            self.L2_sqr = 0
        """

        # negative log likelihood of MLP
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.negative_log_likelihood_sym = (
            self.logRegressionLayer.negative_log_likelihood_sym
        )
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        self.params_sym = []
        for layer in self.hiddenLayer:
            self.params += layer.params
            self.params_sym += layer.params_sym
        self.params += self.logRegressionLayer.params
        self.params_sym += self.logRegressionLayer.params_sym
        # end-snippet-3

        self.n_params = sum(
            [layer_sizes[i]*layer_sizes[i+1] for i in range(len(layer_sizes) - 1)]
        )

        # accumulators
        # self.accs = self.hiddenLayer.accs + self.logRegressionLayer.accs

        # keep track of model input
        self.input = input
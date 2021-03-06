#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # forward propagation
    examples = data.shape[0]
    z2 = np.matmul(data, W1) + b1
    assert z2.shape == (examples, H)

    a2 = sigmoid(z2)
    assert a2.shape == (examples, H)

    z3 = np.matmul(a2, W2) + b2
    assert z3.shape == (examples, Dy)

    y = softmax(z3)
    assert y.shape == (examples, Dy)

    partial_cost = -np.sum(labels * np.log(y), axis=1)
    assert partial_cost.shape == (examples,)
    cost = np.average(partial_cost)

    ### backward propagation
    d3_per_example = (y - labels) / examples
    assert d3_per_example.shape == (examples, Dy)

    # second layer
    gradW2 = np.matmul(np.transpose(a2), d3_per_example)
    assert gradW2.shape == (H, Dy)

    gradb2 = np.sum(d3_per_example, axis=0)
    assert gradb2.shape == (Dy,)

    da2 = np.matmul(d3_per_example, np.transpose(W2))
    assert da2.shape == (examples, H)

    dz2 = sigmoid_grad(a2) * da2
    assert dz2.shape == (examples, H)

    # first layer
    gradW1 = np.matmul(np.transpose(data), dz2)
    assert gradW1.shape == (Dx, H)

    gradb1 = np.sum(dz2, axis=0)
    assert gradb1.shape == (H,)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

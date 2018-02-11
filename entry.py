#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ffnet import FFNet
from layers import FCLayer
from activations import (
    SigmoidActivationFunction, LinearActivationFunction, ReluActivationFunction
)
import numpy as np
from numpy import array, matrix
from autoencoder import Autoencoder
from utils import flatten, randmatrix, randvector

def gen_net(input_dims, mid_layer_dims, n_layers, activation_function):
    layers_dims_half = [input_dims]
    layers_dims = layers_dims_half + [mid_layer_dims]
    layers_dims.extend(reversed(layers_dims_half))
    total_n_weights = 0
    layers = []
    for indim, outdim in zip(layers_dims[:-1], layers_dims[1:]):
        total_n_weights += indim * outdim
        layers.append(FCLayer(shape=(indim, outdim), afun=activation_function))

    weights = []
    for l in layers[:len(layers) // 2]:
        w = randmatrix(l.shape[1], l.shape[0])
        weights.append(w)
        l.set_weights(flatten(w))

    for l, w in zip(layers[len(layers) // 2:], reversed(weights)):
        l.set_weights(flatten(w.T))

    net = FFNet(layers)
    return net, weights

def gen_inputs(input_dims, batch_size):
    return randmatrix(input_dims, batch_size)

def make_layer(weights):
    layer = FCLayer(tuple(reversed(weights.shape)),
                    LinearActivationFunction(koef=1, base=0))
    layer.set_weights(flatten(weights))
    return layer


if __name__ == "__main__":
    input_dims = 4
    mid_layer_dims = 2
    n_layers = 3
    batch_size = 1
    activation_function = LinearActivationFunction()

    # net, weights = gen_net(input_dims, mid_layer_dims, n_layers, activation_function)
    # net.print_weights()
    weights = [matrix('1 2 1; 3 4 2')]
    weights.append(weights[0].T)
    layers = [
        FCLayer((4, 3), LinearActivationFunction()),
        FCLayer((3, 2), LinearActivationFunction()),
        FCLayer((2, 3), LinearActivationFunction()),
        FCLayer((3, 4), LinearActivationFunction()),
    ]
    layers[0].set_weights(matrix('.5 0 0 0;0 .5 0 0;0 0 .5 0'))
    layers[1].set_weights(matrix('.2 0 0;0 .1 0'))
    layers[2].set_weights(matrix('5 0;0 10;5 0'))
    layers[3].set_weights(matrix('2 0 0;0 2 0;2 0 0;0 2 0'))
    net = FFNet(layers)

    # inputs = gen_inputs(input_dims, batch_size)
    m = matrix([
        [7, 9, 4, 1, 1, 0, 3, 3, 5, 6, 5, 7, 2, 9, 1, 4, 6, 3, 6, 7],
        [3, 6, 4, 3, 6, 6, 9, 2, 2, 6, 0, 4, 5, 2, 8, 5, 1, 4, 2, 1]
    ])
    inputs = np.concatenate((m, m))[:,:10]

    print("inputs")
    print(inputs)
    print("outputs")
    outputs = net.compute_outputs(inputs)
    print(outputs)

    print('activations')
    for l in net.layers:
        print(l.activations)

    ac = Autoencoder(net.layers)
    loss, loss_grad = ac.compute_loss(inputs)
    print("loss")
    print(loss)
    print("loss_grad")
    print(loss_grad)

    print("loss accuracy")
    vector_p = array([.1, .05, .07, .1, .05, .07, .1, .1, .05, .05, .07, .07])
    vector_p = array([100,500,700, 100,500,700, 100, 100,500,500,700,700])
    vector_p = array([1000000] * len(net.get_weights()))
    print("vector_p")
    print(vector_p)
    print(ac.compute_loss_grad_accuracy(inputs, vector_p))
    print(ac.compute_loss_grad_accuracy_imag(inputs, vector_p))

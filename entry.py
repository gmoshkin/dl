#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ffnet import FFNet
from layers import FCLayer
from activations import (
    SigmoidActivationFunction, LinearActivationFunction, ReluActivationFunction
)
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
    net = FFNet([make_layer(w) for w in weights])

    # inputs = gen_inputs(input_dims, batch_size)
    inputs = matrix('3;0;-1')

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
    print("vector_p")
    print(vector_p)
    print(ac.compute_loss_grad_accuracy(inputs, vector_p))


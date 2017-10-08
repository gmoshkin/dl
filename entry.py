#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ffnet import FFNet
from layers import FCLayer
from activations import (
    SigmoidActivationFunction, LinearActivationFunction, ReluActivationFunction
)
from numpy import array

def randlist(length, lo=0, hi=10):
    '''returns a list random ints of given length'''
    import random
    return [random.randint(lo, hi) for _ in range(length)]

def matrix(values, n_rows, n_cols):
    import numpy
    return numpy.matrix(values).reshape(n_rows, n_cols)

def randmatrix(n_rows, n_cols):
    return matrix(randlist(n_rows * n_cols), n_rows, n_cols)

def gen_net(input_dims, mid_layer_dims, n_layers, activation_function):
    layers_dims_half = [input_dims]
    layers_dims = layers_dims_half + [mid_layer_dims]
    layers_dims.extend(reversed(layers_dims_half))
    total_n_weights = 0
    layers = []
    for indim, outdim in zip(layers_dims[:-1], layers_dims[1:]):
        total_n_weights += indim * outdim
        layers.append(FCLayer(shape=(indim, outdim), afun=activation_function))
    net = FFNet(layers)
    net.set_weights(array(randlist(total_n_weights)))
    return net

def gen_inputs(input_dims, batch_size):
    return randmatrix(input_dims, batch_size)


if __name__ == "__main__":
    input_dims = 4
    mid_layer_dims = 2
    n_layers = 3
    batch_size = 4
    activation_function = SigmoidActivationFunction()

    net = gen_net(input_dims, mid_layer_dims, n_layers, activation_function)
    net.print_weights()

    inputs = gen_inputs(input_dims, batch_size)

    print("inputs")
    print(inputs)
    print("outputs")
    print(net.compute_outputs(inputs))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ffnet import FFNet
from layers import FCLayer
from activations import (
    SigmoidActivationFunction, LinearActivationFunction, ReluActivationFunction
)
from numpy import array
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


if __name__ == "__main__":
    input_dims = 4
    mid_layer_dims = 2
    n_layers = 3
    batch_size = 4
    activation_function = ReluActivationFunction()

    net, weights = gen_net(input_dims, mid_layer_dims, n_layers, activation_function)
    net.print_weights()

    inputs = gen_inputs(input_dims, batch_size)

    print("inputs")
    print(inputs)
    print("outputs")
    outputs = net.compute_outputs(inputs)
    print(outputs)

    ac = Autoencoder(net.layers)
    loss, loss_grad = ac.compute_loss(inputs)
    print("loss")
    print(loss)
    print("loss_grad")
    print(loss_grad)

    print("loss accuracy")
    vector_p = randvector(sum(w.size for w in weights) * 2)
    print(ac.compute_loss_grad_accuracy(inputs, vector_p))


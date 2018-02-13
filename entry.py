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
from utils import flatten, randmatrix, randvector, lerp, show_digit
from data import read_mnist, get_inputs
import sys
import torch as tch
import torch.nn as tnn
import torch.autograd as tad

NUM_EPOCH = 1000
MINIBATCH_SIZE=200
N_LAYERS = 4

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

class TorchModel(tnn.Module):

    def __init__(self, my_layers):
        super(TorchModel, self).__init__()
        self.layers = []
        for i, ml in enumerate(my_layers):
            new_layer = tnn.Linear(*ml.shape, bias=False)
            self.layers.append(new_layer)
            self.__setattr__('layer_{}'.format(i), new_layer)

    def forward(self, inputs):
        curr_out = inputs
        for l in self.layers:
            curr_out = l(curr_out)
        return curr_out

def train(inputs):
    # TODO: compare with what pytorch does
    num_features = inputs.shape[0]
    middle_size = 2
    half_num_layers = N_LAYERS // 2
    def make_layer(size_in, size_out):
        return FCLayer((size_in, size_out), LinearActivationFunction())
        # return (size_in, size_out)
    layers = []
    for i in range(half_num_layers):
        lesser = int(lerp(middle_size, num_features, i, half_num_layers))
        greater = int(lerp(middle_size, num_features, i + 1, half_num_layers))
        layers = [make_layer(greater, lesser)] + layers + [make_layer(lesser, greater)]
    ae = Autoencoder(layers)
    ae.init_weights()
    print('Network has {} parameters'.format(ae.net.params_number))

    digit_in = inputs[:,0].reshape(-1, 1)
    digit_size = int(np.sqrt(digit_in.size))
    show_digit(digit_in.reshape(digit_size, digit_size))

    print('my net before:')
    my_out = ae.net.compute_outputs(digit_in)
    show_digit(my_out.reshape(digit_size, digit_size))

    tnet = TorchModel(ae.net.layers)
    for p, l in zip(tnet.parameters(), ae.net.layers):
        l.set_weights(p.data.numpy().flatten())

    print('my net after:')
    my_out = ae.net.compute_outputs(digit_in)
    show_digit(my_out.reshape(digit_size, digit_size))

    print('torch net:')
    torch_out = tnet(tad.Variable(tch.Tensor(digit_in.T))).data.numpy()
    show_digit(torch_out.reshape(digit_size, digit_size))

    torch_sgd(tnet, inputs, num_epoch=NUM_EPOCH, minibatch_size=MINIBATCH_SIZE,
              display=True)
    # ae.run_sgd(inputs, num_epoch=NUM_EPOCH, minibatch_size=MINIBATCH_SIZE,
    #            display=True)

def torch_sgd(net, inputs, num_epoch=NUM_EPOCH, minibatch_size=MINIBATCH_SIZE,
              display=True):
    optimizer = tch.optim.SGD(net.parameters(), lr=.01)
    import time, utils
    start_time = time.time()

    def count_loss(inputs, outputs):
        return ((inputs - outputs) ** 2).sum() / 2 / inputs.size()[1]
    def to_torch(matrix):
        return tad.Variable(tch.Tensor(matrix))

    for epoch in range(num_epoch):
        minibatch = utils.get_random_sample(inputs, minibatch_size)
        torch_minibatch = to_torch(minibatch.T)
        optimizer.zero_grad()
        outputs = net(torch_minibatch)
        loss = count_loss(torch_minibatch, outputs)
        loss.backward()

        if epoch % 20 == 0:
            digit_in = minibatch[:,0]
            digit_size = int(np.sqrt(digit_in.size))
            print("input:")
            utils.show_digit(digit_in.reshape(digit_size, digit_size))
            digit_out = net(to_torch(digit_in.T)).data.numpy()
            print("output:")
            utils.show_digit(digit_out.reshape(digit_size, digit_size))

        optimizer.step()

        if display:
            print("[{e}] loss: {l:0.4f} time: {t:0.2f}s".format(
                l=loss.data[0],
                e=epoch,
                t=(time.time() - start_time)
            ))

def test_grad():
    layer = FCLayer((3, 3), LinearActivationFunction())
    layer.set_weights(matrix('1 2 4;2 1 3;1 3 1'))
    net = FFNet([layer])
    inputs = matrix('1;1;1')
    print('inputs:', inputs)
    outputs = net.compute_outputs(inputs)
    print('outputs:', outputs)
    loss_derivs = outputs - inputs
    print('loss_derivs:', loss_derivs)
    loss_grad = net.compute_loss_grad(loss_derivs)
    print('loss_grad:', loss_grad)

def test_data(data):
    index = np.random.randint(data[0].shape[0])
    show_digit(data[0][index])
    print(data[1][index])

if __name__ == "__main__":
    data = read_mnist('mnist', normalize=True)
    # test_data(data)

    train(get_inputs(data[0]))
    # test_grad()

    sys.exit(0)

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

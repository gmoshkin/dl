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
from data import read_mnist, get_inputs, get_digits
import sys
import torch as tch
import torch.nn as tnn
import torch.autograd as tad
import utils

NUM_EPOCH = 1000
MINIBATCH_SIZE=200
N_LAYERS = 4
STEP_SIZE = 0.01
DO_TORCH = True
DO_MINE = False
MIDDLE_SIZE = 10
ACTIVATIONS = 'sigmoid'
# ACTIVATIONS = 'relu'
# ACTIVATIONS = 'linear'
OPTIM = 'sgd' # 'adam' 'rms'

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
        self.activations = []
        for i, ml in enumerate(my_layers):
            new_layer = tnn.Linear(*ml.shape, bias=True)
            self.layers.append(new_layer)
            self.__setattr__('layer_{}'.format(i), new_layer)
            if ACTIVATIONS == 'sigmoid':
                new_activation = tnn.Sigmoid()
            elif ACTIVATIONS == 'relu':
                new_activation = tnn.ReLU()
            else:
                new_activation = lambda x: x
            self.activations.append(new_activation)
            self.__setattr__('activation_{}'.format(i), new_activation)

    def forward(self, inputs):
        curr_out = inputs
        for l, f in zip(self.layers, self.activations):
            curr_out = f(l(curr_out))
        return curr_out

def to_torch(matrix):
    return tad.Variable(tch.Tensor(matrix))

def train(inputs, digits):
    # TODO: compare with what pytorch does
    num_features = inputs.shape[0]
    middle_size = int(MIDDLE_SIZE)
    half_num_layers = int(N_LAYERS) // 2
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
    print('min val', min([np.min(l.get_weights()) for l in ae.net.layers]))
    print('max val', max([np.max(l.get_weights()) for l in ae.net.layers]))

    # digit_in = inputs[:,0].reshape(-1, 1)
    # digit_size = int(np.sqrt(digit_in.size))
    # show_digit(digit_in.reshape(digit_size, digit_size))

    # print('my net before:')
    # my_out = ae.net.compute_outputs(digit_in)
    # show_digit(my_out.reshape(digit_size, digit_size))

    tnet = TorchModel(ae.net.layers)
    # for p, l in zip(tnet.parameters(), ae.net.layers):
    #     l.set_weights(p.data.numpy().flatten())

    print('min val', min([np.min(l.get_weights()) for l in ae.net.layers]))
    print('max val', max([np.max(l.get_weights()) for l in ae.net.layers]))

    # print('my net after:')
    # my_out = ae.net.compute_outputs(digit_in)
    # show_digit(my_out.reshape(digit_size, digit_size))

    # print('torch net:')
    # torch_out = tnet(tad.Variable(tch.Tensor(digit_in.T))).data.numpy()
    # show_digit(torch_out.reshape(digit_size, digit_size))

    torch_loss, my_loss = torch_sgd(ae, tnet, inputs, num_epoch=NUM_EPOCH,
                                    minibatch_size=MINIBATCH_SIZE,
                                    display=True)

    for digit_in in digits:
        digit_in = digit_in.reshape((-1, 1))
        digit_size = int(np.sqrt(digit_in.size))
        print("input:")
        utils.show_digit(digit_in.reshape(digit_size, digit_size))

        if int(DO_TORCH):
            digit_out = tnet(to_torch(digit_in.T)).data.numpy()
            print("torch output:")
            utils.show_digit(digit_out.reshape(digit_size, digit_size))

        if int(DO_MINE):
            digit_out = ae.net.compute_outputs(digit_in)
            print("my output:")
            utils.show_digit(digit_out.reshape(digit_size, digit_size))
        print('my loss: {:0.4f} torch loss: {:0.4f}'.format(my_loss,
                                                            torch_loss))

    # ae.run_sgd(inputs, num_epoch=int(NUM_EPOCH), minibatch_size=int(MINIBATCH_SIZE),
    #            display=True)

def torch_sgd(ae, net, inputs, num_epoch=int(NUM_EPOCH),
              minibatch_size=int(MINIBATCH_SIZE), display=True):
    momentum = 0.9
    if OPTIM == 'sgd':
        optimizer = tch.optim.SGD(net.parameters(), lr=STEP_SIZE)
    elif OPTIM == 'rms':
        optimizer = tch.optim.RMSprop(net.parameters(), lr=STEP_SIZE)
    elif OPTIM == 'adam':
        optimizer = tch.optim.Adam(net.parameters(), lr=STEP_SIZE)
    else:
        raise Exception('unknown optimization strat: "{}"'.format(OPTIM))
    import time
    start_time = time.time()

    def count_loss(inputs, outputs):
        return ((inputs - outputs) ** 2).sum() / 2 / inputs.size()[1]

    velocity = np.zeros(ae.net.params_number)
    torch_loss = -1
    train_loss = -1

    for epoch in range(num_epoch):
        minibatch = utils.get_random_sample(inputs, minibatch_size)

        if int(DO_TORCH):
            # torch
            torch_minibatch = to_torch(minibatch.T)
            optimizer.zero_grad()
            outputs = net(torch_minibatch)
            loss = count_loss(torch_minibatch, outputs)
            loss.backward()
            torch_loss = loss.data[0]

            if epoch % 20 == 0:
                digit_in = minibatch[:,0]
                digit_size = int(np.sqrt(digit_in.size))
                print("input:")
                utils.show_digit(digit_in.reshape(digit_size, digit_size))
                digit_out = net(to_torch(digit_in.T)).data.numpy()
                print("torch output:")
                utils.show_digit(digit_out.reshape(digit_size, digit_size))

            optimizer.step()

        if int(DO_MINE):
            # mine
            old_weights = ae.net.get_weights()
            # θ ̃← θ + αv
            ae.net.set_weights(old_weights + momentum * velocity)
            # L(x, f(x|θ ̃)), ∇_(θ ̃)L
            train_loss, train_grad = ae.compute_loss(minibatch)
            train_grad = train_grad / np.linalg.norm(train_grad)
            # v ← αv − εg
            velocity = momentum * velocity - STEP_SIZE * train_grad
            # θ ← θ + v
            new_weights = old_weights + velocity

            if epoch % 20 == 0:
                digit_in = minibatch[:,0]
                digit_size = int(np.sqrt(digit_in.size))
                print("input:")
                utils.show_digit(digit_in.reshape(digit_size, digit_size))
                digit_out = ae.net.compute_outputs(digit_in)
                print("my output:")
                utils.show_digit(digit_out.reshape(digit_size, digit_size))

            ae.net.set_weights(new_weights)

        if display:
            print("[{e}] torch loss: {tl:0.4f} my loss: {ml:0.4f} ratio: {lr:0.4f} time: {t:0.2f}s".format(
                tl=torch_loss,
                ml=train_loss,
                lr=torch_loss / (train_loss or 1),
                e=epoch,
                t=(time.time() - start_time)
            ))
    return torch_loss, train_loss

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
    for arg in sys.argv[1:]:
        if arg.count('=') == 1:
            var, val = arg.split('=')
            globals()[var] = val

    NUM_EPOCH = int(NUM_EPOCH)
    MINIBATCH_SIZE = int(MINIBATCH_SIZE)
    N_LAYERS = int(N_LAYERS)
    STEP_SIZE = float(STEP_SIZE)
    DO_TORCH = bool(int(DO_TORCH))
    DO_MINE = bool(int(DO_MINE))
    MIDDLE_SIZE = int(MIDDLE_SIZE)

    data = read_mnist('mnist', normalize=True)
    # test_data(data)

    train(get_inputs(data[0]), get_digits(data[0], data[1]))
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
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
import datetime

NUM_EPOCH = 64
MINIBATCH_SIZE=50
N_LAYERS = 4
STEP_SIZE = 0.01
DO_TORCH = False
DO_MINE = True
MIDDLE_SIZE = 2
ACTIVATIONS = 'linear'
# ACTIVATIONS = 'relu'
# ACTIVATIONS = 'linear'
OPTIM = 'adam' # 'sgd' 'adam' 'rms'
SHOW_DIGITS = False
WEIGHT_DECAY = .4
USE_BIAS = False
VIS_DIGITS = False
VIS_MID_LAYER = False
COPY_TORCH_WEIGHTS = False

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
        self.layer_outputs = []
        for i, ml in enumerate(my_layers):
            new_layer = tnn.Linear(*ml.shape, bias=USE_BIAS)
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
        self.layer_outputs = [inputs]
        for l, f in zip(self.layers, self.activations):
            self.layer_outputs.append(f(l(self.layer_outputs[-1])))
        return self.layer_outputs[-1]

    def decode(self, inputs):
        curr_out = inputs
        for l, f in zip(self.layers[N_LAYERS // 2:],
                        self.activations[N_LAYERS // 2:]):
            curr_out = f(l(curr_out))
        return curr_out

def to_torch(matrix):
    return tad.Variable(tch.Tensor(matrix))

def get_globals():
    import re
    regex = re.compile('^[A-Z_]+$')
    res = []
    for k, v in globals().items():
        if regex.match(k):
            res.append("{}={}".format(k, v))
    return ', '.join(res)

def gen_layers(num_features):
    middle_size = MIDDLE_SIZE
    half_num_layers = N_LAYERS // 2
    def make_layer(size_in, size_out):
        return FCLayer((size_in, size_out), LinearActivationFunction())
    layers = []
    for i in range(half_num_layers):
        lesser = int(lerp(middle_size, num_features, i / half_num_layers))
        greater = int(lerp(middle_size, num_features, (i + 1) / half_num_layers))
        layers = [make_layer(greater, lesser)] + layers + [make_layer(lesser, greater)]
    return layers

def copy_weights(tnet, ae):
    for p, l in zip(tnet.parameters(), ae.net.layers):
        l.set_weights(p.data.numpy().flatten())

def train(inputs, validation, digits):
    # TODO: compare with what pytorch does
    ae = Autoencoder(gen_layers(inputs.shape[0]))
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

    if COPY_TORCH_WEIGHTS:
        copy_weights(tnet, ae)

    print('min val', min([np.min(l.get_weights()) for l in ae.net.layers]))
    print('max val', max([np.max(l.get_weights()) for l in ae.net.layers]))

    # print('my net after:')
    # my_out = ae.net.compute_outputs(digit_in)
    # show_digit(my_out.reshape(digit_size, digit_size))

    # print('torch net:')
    # torch_out = tnet(tad.Variable(tch.Tensor(digit_in.T))).data.numpy()
    # show_digit(torch_out.reshape(digit_size, digit_size))

    stats = torch_sgd(
        ae, tnet, inputs, validation, num_epoch=NUM_EPOCH,
        minibatch_size=MINIBATCH_SIZE, display=True
    )

    digit_size = int(np.sqrt(ae.net.layers[0].shape[0]))

    if VIS_DIGITS:
        visualize_digits(tnet, ae, digits, digit_size)

    date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    if VIS_MID_LAYER:
        visualize_middle_layer(tnet, digit_size,
                               filename='visualization_{}.png'.format(date_string))

    save_stats(stats, filename='stats_{}'.format(date_string))

    # ae.run_sgd(inputs, num_epoch=int(NUM_EPOCH), minibatch_size=int(MINIBATCH_SIZE),
    #            display=True)


def visualize_digits(tnet, ae, digits, digit_size):
    for digit_in in digits:
        digit_in = digit_in.reshape((-1, 1))
        digit_size = int(np.sqrt(digit_in.size))
        print("input:")
        utils.show_digit(digit_in.reshape(digit_size, digit_size))

        if int(DO_TORCH):
            digit_out = tnet(to_torch(digit_in.T)).data.numpy()
            print("torch output:")
            utils.show_digit(digit_out.reshape(digit_size, digit_size))
            print("torch middle layer:", tnet.layer_outputs[N_LAYERS // 2])

        if int(DO_MINE):
            digit_out = ae.net.compute_outputs(digit_in)
            print("my output:")
            utils.show_digit(digit_out.reshape(digit_size, digit_size))
            print("mid layer out:", ae.net.layers[N_LAYERS // 2 - 1].activations.flatten())

def visualize_middle_layer(tnet, digit_size, filename):
    left, right = -2, 2
    steps = 40
    visualization = []
    for i in range(steps):
        row = []
        y = lerp(left, right, i/(steps - 1))
        for j in range(steps):
            # print('decoding ', one, two)
            x = lerp(left, right, j/(steps - 1))
            # print('[{:0.3f}, {:0.3f}]'.format(x, y), end=' ')
            out = tnet.decode(to_torch(matrix((x, y)))).data.numpy()
            row.append(out.reshape(digit_size, digit_size))
        # utils.show_digit(np.concatenate(row, axis=1))
        visualization.append(np.concatenate(row, axis=1))
    vis = np.concatenate(visualization, axis=0)

    utils.save_image(vis, filename)

def save_stats(stats, filename):
    stat = get_globals()
    stat += '\nmine train: {mt:0.4f} test: {mv:0.4f} torch train: {tt:0.4f} test: {tv:0.4f} time spent: {t:0.2f}'.format(
        mt=stats['my_train_loss'],
        tt=stats['torch_train_loss'],
        t=stats['time'],
        mv=stats['my_test_loss'],
        tv=stats['torch_test_loss']
    )
    with open(filename, 'w') as f:
        print(stat, file=f)
    print(stat)

def count_loss(inputs, outputs):
    return ((inputs - outputs) ** 2).sum() / 2 / inputs.size()[0]

torch_outputs = None
my_outputs = None

def torch_sgd(ae, net, inputs, validation=None, num_epoch=int(NUM_EPOCH),
              minibatch_size=int(MINIBATCH_SIZE), display=False):
    momentum = 0.9
    if OPTIM == 'sgd':
        optimizer = tch.optim.SGD(net.parameters(), lr=STEP_SIZE,
                                  weight_decay=WEIGHT_DECAY)
    elif OPTIM == 'rms':
        optimizer = tch.optim.RMSprop(net.parameters(), lr=STEP_SIZE,
                                      weight_decay=WEIGHT_DECAY)
    elif OPTIM == 'adam':
        optimizer = tch.optim.Adam(net.parameters(), lr=STEP_SIZE,
                                   weight_decay=WEIGHT_DECAY)
    else:
        raise Exception('unknown optimization strat: "{}"'.format(OPTIM))
    import time
    start_time = time.time()

    velocity = np.zeros(ae.net.params_number)
    torch_loss = -1
    train_loss = -1
    torch_test_loss = -1
    test_loss = -1

    loss_func = tnn.MSELoss()
    def count_loss(inputs, outputs):
        print(inputs.size(), outputs.size())
        return loss_func(inputs, outputs)

    for epoch in range(num_epoch):
        minibatch = utils.get_random_sample(inputs, minibatch_size)
        if validation is not None:
            valid_batch = utils.get_random_sample(validation, minibatch_size)

        if int(DO_TORCH):
            # torch
            torch_minibatch = to_torch(minibatch.T)
            optimizer.zero_grad()
            outputs = net(torch_minibatch)
            loss = count_loss(outputs, torch_minibatch)
            loss.backward()
            torch_loss = loss.data[0]

            torch_valid = to_torch(valid_batch.T)
            torch_test_loss = count_loss(net(torch_valid), torch_valid).data[0]

            if epoch % 20 == 0 and SHOW_DIGITS:
                digit_in = valid_batch[:,0]
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
            test_loss = np.power(minibatch - ae.net.layers[-1].activations, 2).sum() / 2 / MINIBATCH_SIZE

            # train_grad = train_grad / np.linalg.norm(train_grad)
            # v ← αv − εg
            velocity = momentum * velocity - STEP_SIZE * train_grad
            # θ ← θ + v
            new_weights = old_weights + velocity

            if epoch % 20 == 0 and SHOW_DIGITS:
                digit_in = valid_batch[:,0]
                digit_size = int(np.sqrt(digit_in.size))
                print("input:")
                utils.show_digit(digit_in.reshape(digit_size, digit_size))
                digit_out = ae.net.compute_outputs(digit_in)
                print("my output:")
                utils.show_digit(digit_out.reshape(digit_size, digit_size))

            ae.net.set_weights(new_weights)

        if display:
            print("[{e}] torch train: {tl:0.3f} test: {tv:0.3f} my train: {ml:0.3f} test: {mv:0.3f} ratio: {lr:0.3f} time: {t:0.2f}s".format(
                tl=torch_loss, tv=torch_test_loss,
                ml=train_loss, mv=test_loss,
                lr=torch_loss / (train_loss or 1),
                e=epoch,
                t=(time.time() - start_time)
            ))

    return {
        'torch_train_loss' : torch_loss,
        'torch_test_loss'  : torch_test_loss,
        'my_train_loss'    : train_loss,
        'my_test_loss'     : test_loss,
        'time'             : (time.time() - start_time),
    }

def test_grad():
    layer = FCLayer((3, 3), LinearActivationFunction())
    layer.set_weights(matrix('1 2 4;2 1 3;1 3 1'))
    net = FFNet([layer])

    class Tmp(tnn.Module):
        def __init__(self, my_layer):
            super(Tmp, self).__init__()
            self.layer = tnn.Linear(*my_layer.shape, bias=False)

        def forward(self, inputs):
            self.outputs = self.layer(inputs)
            return self.outputs

    tnet = Tmp(layer)
    net.layers[0].set_weights(next(tnet.parameters()).data.numpy().flatten())

    inputs = matrix('1;1;1')
    print('inputs:', inputs)
    outputs = net.compute_outputs(inputs)
    print('outputs:', outputs)
    loss_derivs = outputs - inputs
    print('loss_derivs:', loss_derivs)
    loss_grad = net.compute_loss_grad(loss_derivs)
    print('loss_grad:', loss_grad)

    t_inputs = to_torch(inputs.T)
    t_outputs = tnet(t_inputs)
    print('t_outputs:', t_outputs)
    t_loss = count_loss(t_inputs, t_outputs)
    print('t_loss:', t_loss)
    t_loss.backward()
    print('loss_grad:', t_loss.grad)
    print('loss_grad:', t_outputs.grad)
    print('loss_grad:', t_inputs.grad)





def test_data(data):
    index = np.random.randint(data[0].shape[0])
    show_digit(data[0][index])
    utils.save_image(data[0][index], 'test.png')
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
    WEIGHT_DECAY = float(WEIGHT_DECAY)
    USE_BIAS = bool(int(USE_BIAS))
    VIS_DIGITS = bool(int(VIS_DIGITS))
    VIS_MID_LAYER = bool(int(VIS_MID_LAYER))
    COPY_TORCH_WEIGHTS = bool(int(COPY_TORCH_WEIGHTS))

    data = read_mnist('/home/gmoshkin/cmc/dl/autoencoder/mnist', normalize=True)
    # test_data(data)


    train(get_inputs(data[0]), get_inputs(data[2]),
          get_digits(data[0], data[1]))

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

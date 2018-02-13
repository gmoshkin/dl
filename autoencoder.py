#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

import ffnet
from numpy.linalg import norm
from numpy import dot, array, r_
import numpy as np
from sys import float_info
from utils import flatten, get_random_sample, compute_norm
import utils
import time

class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and outputs is different!')

        self.loss = None
        self.loss_grad = None

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of data
        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """
        new_output = self.net.compute_outputs(inputs)
        # print("new_output:", new_output, 'shape:', new_output.shape)

        diff = new_output - inputs
        # print("diff:", diff, 'shape:', diff.shape)
        self.loss = compute_norm(diff)
        # print("loss:", self.loss)
        self.net.compute_loss_grad(diff)
        loss_grad = array([])
        for l1, l2 in zip(self.net.layers, reversed(self.net.layers)):
            loss_grad = r_[loss_grad,
                           flatten(l1.w_derivs + l2.w_derivs.T)]
        # print("loss_grad raw")
        # for l in self.net.layers:
            # print(l.w_derivs, 'shape:', l.w_derivs.shape)
        self.loss_grad = loss_grad
        return self.loss, self.loss_grad

    def compute_loss_grad_accuracy(self, inputs, vector_p, eps=0.00000001):
        if self.loss is not None and self.loss_grad is not None:
            loss, loss_grad = self.loss, self.loss_grad
        else:
            loss, loss_grad = self.compute_loss(inputs)
        ofs_weights = self.net.get_weights() + vector_p * eps
        # print("offset weights")
        # print(ofs_weights)
        self.net.set_weights(ofs_weights)
        loss_ofs, _ = self.compute_loss(inputs)
        # print("loss_grad:", loss_grad)
        lhs = dot(loss_grad, vector_p)
        # print("dot(loss_grad, vector_p):", lhs)
        rhs = (loss_ofs - loss) / eps
        # print("loss_ofs:", loss_ofs)
        # print("loss:", loss)
        # print("(loss_ofs - loss) / eps:", rhs)
        return lhs - rhs

    def compute_loss_grad_accuracy_imag(self, inputs, vector_p, eps=0.00000001):
        if self.loss is not None and self.loss_grad is not None:
            loss, loss_grad = self.loss, self.loss_grad
        else:
            loss, loss_grad = self.compute_loss(inputs)
        ofs_weights = self.net.get_weights() + vector_p * eps * 1j
        # print("offset weights")
        # print(ofs_weights)
        self.net.set_weights(ofs_weights)
        loss_ofs, _ = self.compute_loss(inputs)
        # print("loss_grad:", loss_grad)
        lhs = dot(loss_grad, vector_p)
        # print("dot(loss_grad, vector_p):", lhs)
        rhs = (loss_ofs.imag) / eps
        # print("loss_ofs:", loss_ofs)
        # print("(loss_ofs.imag) / eps:", rhs)
        return lhs - rhs



    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """
        pass

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """
        pass

    def run_sgd(self, inputs, step_size=0.01, momentum=0.9, num_epoch=200,
                minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                display=False):
        """
        Stochastic gradient descent optimization
        :param inputs: training sample, numpy matrix of size num_features x
            num_objects
        :param step_size: step size, number
        :param momentum: momentum coefficient, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x
            num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the
                following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each
                epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each
                peoch, list
        """
        velocity = np.zeros(self.net.params_number)
        start_time = time.time()
        for epoch in range(num_epoch):
            minibatch = get_random_sample(inputs, minibatch_size)

            old_weights = self.net.get_weights()
            # θ ̃← θ + αv
            self.net.set_weights(old_weights + momentum * velocity)
            # L(x, f(x|θ ̃)), ∇_(θ ̃)L
            train_loss, train_grad = self.compute_loss(minibatch)
            train_grad = train_grad / norm(train_grad)
            # v ← αv − εg
            velocity = momentum * velocity - step_size * train_grad
            # θ ← θ + v
            new_weights = old_weights + velocity

            if epoch % 20 == 0:
                digit_in = minibatch[:,0]
                digit_size = int(np.sqrt(digit_in.size))
                print("input:")
                utils.show_digit(digit_in.reshape(digit_size, digit_size))
                digit_out = self.net.compute_outputs(digit_in)
                print("output:")
                utils.show_digit(digit_out.reshape(digit_size, digit_size))

            self.net.set_weights(new_weights)

            if display:
                print("[{e}] loss: {l:0.4f} time: {t:0.2f}s".format(
                    l=train_loss,
                    e=epoch,
                    t=(time.time() - start_time)
                ))

        return train_loss, train_grad, 0, np.zeros(self.net.params_number)

    def run_rmsprop(self, inputs, step_size=0.01, num_epoch=200, minibatch_size=100, l2_coef=1e-5,
                 test_inputs=None, display=False):
        """
        RMSprop stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        pass

    def run_adam(self, inputs, step_size=0.01, num_epoch=200, minibatch_size=100, l2_coef=1e-5,
                 test_inputs=None, display=False):
        """
        ADAM stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        pass

    def init_weights(self):
        for l in self.net.layers:
            l.set_weights(np.random.rand(l.get_params_number()))

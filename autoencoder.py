#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

import ffnet
from numpy.linalg import norm
from numpy import dot, array, r_
from sys import float_info
from utils import flatten, compute_norm

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
        diff = self.net.compute_outputs(inputs) - inputs
        self.loss = compute_norm(diff) / (2 * inputs.shape[1]) # wtf is this?
        self.net.compute_loss_grad(diff)
        loss_grad = array([])
        for l1, l2 in zip(self.net.layers, reversed(self.net.layers)):
            loss_grad = r_[loss_grad,
                           flatten(l1.w_derivs + l2.w_derivs.T)]
        print("loss_grad raw")
        print(self.net.layers[0].w_derivs)
        print(self.net.layers[1].w_derivs)
        self.loss_grad = loss_grad
        return self.loss, self.loss_grad

    def compute_loss_grad_accuracy(self, inputs, vector_p, eps=0.001):
        if self.loss is not None and self.loss_grad is not None:
            loss, loss_grad = self.loss, self.loss_grad
        else:
            loss, loss_grad = self.compute_loss(inputs)
        ofs_weights = self.net.get_weights() + vector_p * eps
        print("offset weights")
        print(ofs_weights)
        self.net.set_weights(ofs_weights)
        loss_ofs, _ = self.compute_loss(inputs)
        lhs = dot(loss_grad, vector_p)
        print("lhs")
        print(lhs)
        rhs = (loss_ofs - loss) / eps
        print("loss_ofs")
        print(loss_ofs)
        print("loss")
        print(loss)
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

    def run_sgd(self, inputs, step_size=0.01, momentum=0.9, num_epoch=200, minibatch_size=100, l2_coef=1e-5,
                 test_inputs=None, display=False):
        """
        Stochastic gradient descent optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param momentum: momentum coefficient, number
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



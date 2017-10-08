#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import exp, square as sqr, power as pow
# Implementation of activation functions used within neural networks

class BaseActivationFunction(object):

    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')


class LinearActivationFunction(BaseActivationFunction):

    def __init__(self, koef=1, base=0):
        self.koef = koef
        self.base = base

    def val(self, inputs):
        return inputs * self.koef + self.base

    def deriv(self, inputs):
        return self.koef

    def second_deriv(self, inputs):
        return 0


class SigmoidActivationFunction(BaseActivationFunction):

    def val(self, inputs):
        return 1 / (1 + exp(-inputs))

    def deriv(self, inputs):
        exp_inputs = exp(inputs)
        return exp_inputs / (1 + exp_inputs) ** 2

    def second_deriv(self, inputs):
        exp_inputs = exp(inputs)
        return exp_inputs * (exp_inputs - 1) / (1 + exp_inputs) ** 3


class ReluActivationFunction(BaseActivationFunction):

    def val(self, inputs):
        return inputs * (inputs > 0)

    def deriv(self, inputs):
        return 1

    def second_deriv(self, inputs):
        return 0

#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import matrix, array
# Implementation of layers used within neural networks

class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of
        BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.shape = (self.num_inputs, self.num_outputs) = shape
        self.afun = afun
        self.use_bias = use_bias
        self.weights = None

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        return self.num_inputs * self.num_outputs

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        return array(self.weights).ravel()

    def reshape(self, vector):
        return matrix(vector).reshape(self.num_outputs, self.num_inputs)

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer
        parameters in convenient shape, e.g. matrix shape for fully-connected
        layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        self.weights = self.reshape(w)

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it
        to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        self.direction = self.reshape(p)

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within
        layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x
        num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x
        num_objects
        """
        self.linear_parts = self.weights * inputs
        self.activations = self.afun(self.linear_parts)
        return self.activations

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within
        layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of
        size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix
        of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector
        of length num_params
        """
        raise NotImplementedError()

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within
        layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x
        num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size
        num_outputs x num_objects
        """
        raise NotImplementedError()

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix
        of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy
        matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy
        vector of length num_params
        """
        raise NotImplementedError()

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of
        size num_outputs x num_objects
        """
        raise NotImplementedError()

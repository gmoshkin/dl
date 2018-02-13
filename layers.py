#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import to_matrix, flatten
from numpy import multiply
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
        return flatten(self.weights)

    def reshape(self, vector):
        return to_matrix(vector, self.num_outputs, self.num_inputs)

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
        self.prev_activations = inputs # z₍ᵢ₋₁₎
        self.linear_parts = self.weights * inputs # uᵢ
        self.activations = self.afun(self.linear_parts) # zᵢ
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
        # g'(uⁱ)
        self.afun_derivs = self.afun.deriv(self.linear_parts)
        # ∇uⁱL=∇zⁱL⊙ g'(uⁱ)
        self.linear_derivs = multiply(derivs, self.afun_derivs)
        # ∇wⁱL = ∇uⁱL⋅z⁽ⁱ¯¹⁾ᵀ
        self.w_derivs = self.linear_derivs * self.prev_activations.T
        # ∇z⁽ⁱ¯¹⁾L = wⁱᵀ⋅∇uⁱL
        self.input_derivs = self.weights.T * self.linear_derivs
        return self.input_derivs, flatten(self.w_derivs)

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
        return self.activations

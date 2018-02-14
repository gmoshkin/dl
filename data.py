#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mnist
import os
import numpy as np

train_inputs_file = 'train-images-idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte'
test_inputs_file = 't10k-images-idx3-ubyte'
test_labels_file = 't10k-labels-idx1-ubyte'

def read_mnist(mnist_path, normalize=False):
    with open(os.path.join(mnist_path, train_inputs_file), 'rb') as f:
        train_inputs = mnist.parse_idx(f)
    with open(os.path.join(mnist_path, train_labels_file), 'rb') as f:
        train_labels = mnist.parse_idx(f)
    with open(os.path.join(mnist_path, test_inputs_file), 'rb') as f:
        test_inputs = mnist.parse_idx(f)
    with open(os.path.join(mnist_path, test_labels_file), 'rb') as f:
        test_labels = mnist.parse_idx(f)
    if normalize:
        train_inputs = train_inputs / np.max(train_inputs)
        test_inputs = test_inputs / np.max(test_inputs)
    return train_inputs, train_labels, test_inputs, test_labels

def get_inputs(mnist_data):
    # in: shape = [num_objects x object_height x object_width], values in 0..255
    # out: shape = [num_features x num_objects], values in 0..1
    num_objects, object_height, object_width = mnist_data.shape
    return mnist_data.reshape(-1, object_height * object_width).transpose() / np.max(mnist_data)

def get_digits(images, labels):
    digits = []
    for i in range(10):
        for label, image in zip(labels, images):
            if i == label:
                digits.append(image)
                break
    return digits

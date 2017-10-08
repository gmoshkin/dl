#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import random

def randlist(length, lo=0, hi=10):
    '''returns a list random ints of given length'''
    return [random.randint(lo, hi) for _ in range(length)]

def to_matrix(values, n_rows, n_cols):
    return numpy.matrix(values).reshape(n_rows, n_cols)

def randmatrix(n_rows, n_cols):
    return to_matrix(randlist(n_rows * n_cols), n_rows, n_cols)

def flatten(matr):
    return numpy.array(matr).ravel()

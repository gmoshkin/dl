#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import random

lo = -1
hi = 1

def rand(lo=lo, hi=hi):
    return random.random() * (hi - lo) + lo

def randlist(length, lo=lo, hi=hi):
    '''returns a list random ints of given length'''
    return [rand(lo, hi) for _ in range(length)]

def randmatrix(n_rows, n_cols, lo=lo, hi=hi):
    return to_matrix(randlist(n_rows * n_cols, lo, hi), n_rows, n_cols)

def randvector(n_elems, lo=lo, hi=hi):
    return numpy.array(randlist(n_elems, lo, hi))

def to_matrix(values, n_rows, n_cols):
    return numpy.matrix(values).reshape(n_rows, n_cols)

def flatten(matr):
    return numpy.array(matr).ravel()

def compute_norm(matr):
    N = matr.shape[1]
    per_batch = numpy.einsum('ij,ij->j', matr, matr) / 2
    return sum(per_batch) / N

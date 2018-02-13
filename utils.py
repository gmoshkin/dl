#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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
    return np.array(randlist(n_elems, lo, hi))

def to_matrix(values, n_rows, n_cols):
    return np.matrix(values).reshape(n_rows, n_cols)

def flatten(matr):
    return np.array(matr).ravel()

def compute_norm(matr):
    return np.einsum('ij,ij', matr, matr) / 2 / matr.shape[1]

def get_random_sample(inputs, batch_size):
    num_features, num_objects = inputs.shape
    sample_objects = []
    for i in np.random.choice(num_objects, size=batch_size):
        sample_objects.append(inputs[:,i])
    return np.matrix(sample_objects).transpose()

def lerp(left, right, i, N):
    return left * (1 - i * 1/N) + right * (i * 1/N)

def show_digit(digit, #lo=0, hi=255,
               bg=np.array((0, 43, 54)), fg=np.array((147, 161, 161))):
    min_val, max_val = np.min(digit), np.max(digit)
    print("min:", min_val, "max:", max_val)
    normalized = (digit - min_val) / (max_val - min_val)
    # print("new min:", np.min(normalized), "new max:", np.max(normalized))
    def show_cell(cell):
        # cell must be 0 <= cell <= 1
        # perc = (cell - lo) / hi - lo
        r, g, b = (bg * (1 - cell) + fg * cell).astype(int)
        print('\033[48;2;{r};{g};{b}m  \033[0m'.format(r=r, g=g, b=b), end='')
    for row in normalized:
        for cell in np.nditer(row):
            show_cell(cell)
        print()

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

def lerp(left, right, ratio):
    return left * (1 - ratio) + right * ratio

def show_digit(digit, #lo=0, hi=255,
               bg=np.array((0, 43, 54)), fg=np.array((147, 161, 161)),
               pixels=True):
    min_val, max_val = np.min(digit), np.max(digit)
    print("min:", min_val, "max:", max_val)
    normalized = (digit - min_val) / (max_val - min_val)
    # print("new min:", np.min(normalized), "new max:", np.max(normalized))
    def show_cell(cell):
        # cell must be 0 <= cell <= 1
        # perc = (cell - lo) / hi - lo
        r, g, b = (bg * (1 - cell) + fg * cell).astype(int)
        print('\033[48;2;{r};{g};{b}m  \033[0m'.format(r=r, g=g, b=b), end='')
    def show_double_spaces():
        for row in normalized:
            for cell in np.nditer(row):
                show_cell(cell)
            print()
    def show_2cell(top, bot):
        tR, tG, tB = (bg * (1 - top) + fg * top).astype(int)
        bR, bG, bB = (bg * (1 - bot) + fg * bot).astype(int)
        print('\033[48;2;{tR};{tG};{tB}m\033[38;2;{bR};{bG};{bB}m\u2584\033[0m'.format(
            tR=tR, tG=tG, tB=tB, bR=bR, bG=bG, bB=bB
        ), end='')
    def show_pixels():
        for top_row, bot_row in zip(*([iter(normalized)] * 2)):
            for top, bot in zip(np.nditer(top_row), np.nditer(bot_row)):
                show_2cell(top, bot)
            print()
    if pixels:
        show_pixels()
    else:
        show_double_spaces()

def save_image(data, filename):
    from PIL import Image
    min_val, max_val = np.min(data), np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    im = Image.fromarray(np.uint8(normalized*255))
    im.save(filename)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-28-20 23:10
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

"""Implement a XOR gate Neural Network
"""
import numpy as np


def init_weights(layers, epsilon):
    """init_weights
    use numpy's built-in np.random.rand() function to initialize the weights, it samples values from a uniform distribution over [0, 1).
    """
    weights = []
    for i in range(len(layers)-1):
        w = np.random.rand(layers[i+1], layers[i]+1)
        w = epsilon * (2 * w * - 1)
        weights.append(np.mat(w))
    return weights


X = np.mat([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
Y = np.mat([0, 1, 1, 0])
layers = [2, 2, 1]
epochs = 5000
alpha = 0.5
w = init_weights(layers, 1)
result = {'J': [], 'h': []}
w_s = {}


def main():
    pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-28-20 23:06
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)


"""Simple NN layers
"""
import numpy as np


def sigmoid(x):
    """sigmoid function
    # Input
        x: array-like data.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_gradient(z):
    """sigmoid gradient function
    # Input
        z: array-like data.
    """
    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g


def main():
    pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-10-21 17:44
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


def MLP(input_shape, units=2, kernel_initializer='glorot_uniform', bias_initializer='zeros', num_classes=1):
    """Simple MLP network for XOR dataset.
    Inputs:
        input_shape: 
        units: units of hidden layers, default 2, 2 units
    """
    model = Sequential(name="MLP")

    model.add(Input(shape=input_shape))
    model.add(Dense(units, activation=None,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    name="dense_1"))
    model.add(Dense(num_classes, activation='sigmoid',
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    name="output"))

    model.build()

    return model


def main():
    import numpy as np
    X = np.array([[1, 0]])
    y = np.array([1])
    m = MLP(input_shape=(2, ), num_classes=1)
    print(m(X))


if __name__ == "__main__":
    main()

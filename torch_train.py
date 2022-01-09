#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-14-21 01:57
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from xor_gate_nn.models.torch.mlp import MLP
from xor_gate_nn.datasets.torch.datasets import XOR_Dataset


def rand_initialize_weights(L_in, L_out, epsilon):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections;

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    W = epsilon * (2 * np.random.rand(L_in+1, L_out) * - 1)
    return W


def get_output_map(model, N=5):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    output_map = []
    for _x in x:
        for _y in y:
            o = model(np.array([[_x, _y]])).numpy()[0][0]
            output_map.append(o)
    output_map = np.asarray(output_map).reshape((N, N))

    return output_map


def main():
    dataset = XOR_Dataset(batch_size=4)  # Full batch learning

    model_name = "MLP"
    date_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    prefix = os.path.join("~", "Documents", "DeepLearningData", "xor")
    subfix = os.path.join(model_name, date_time)
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))
    ckpt_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # units=3
    # bias_initializer="glorot_uniform"
    # bias_initializer="zeros"
    # bias_initializer="ones"
    from tensorflow.keras.initializers import RandomUniform
    mlp = MLP(input_shape=(2, ), units=2,  # units=2 or 3
              kernel_initializer=RandomUniform(minval=-1.0, maxval=1.0),
              bias_initializer=RandomUniform(minval=-1.0, maxval=1.0),
              num_classes=1)

    # set_weights(mlp)

    mlp.save_weights(os.path.join(ckpt_dir, "initial_weights.hdf5"))

    # tf.keras.losses.MeanAbsoluteError(), x
    # tf.keras.losses.Hinge(), x
    # loss = tf.keras.losses.Huber(), yes
    # loss = tf.keras.losses.CategoricalCrossentropy(), x
    loss = tf.keras.losses.BinaryCrossentropy()

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.99)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    mlp.compile(loss=loss, optimizer=optimizer)

    epochs = 4000
    history = mlp.fit(dataset, epochs=epochs, verbose=1)

    loss = history.history["loss"]
    plt.plot(np.arange(1, epochs+1), loss)
    plt.title("Loss-epoch curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(log_dir, "loss.jpg"))
    plt.show()

    mlp.save_weights(os.path.join(ckpt_dir, "final_weights.hdf5"))

    output_map = get_output_map(mlp, N=20)
    plt.imshow(output_map, origin="lower",
               cmap="rainbow", extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.title("MLP output on x-y plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(log_dir, "output_map.jpg"))
    plt.show()


if __name__ == "__main__":
    main()

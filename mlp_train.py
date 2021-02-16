#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-14-21 01:57
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from xor_gate_nn.models.keras_fn.mlp import MLP
from xor_gate_nn.datasets.keras_fn.datasets import XOR_Dataset


def difference_loss(labels, predictions):
    """
    Computer difference loss
    Args:
      labels: A int32 tensor of shape [batch_size].
      predictions: A float32 tensor of shape [batch_size,num_classes].
    Returns:
      A tensor of the same shape as `lables`
    """
    L = predictions - labels
    L = tf.reduce_sum(L, axis=1)
    return L


class DifferenceLoss(tf.keras.losses.Loss):
    """Implements a difference

    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `BinaryFocalLoss`.

        Arguments:
        """
        super(DifferenceLoss, self).__init__(
            reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Invokes the `DifferenceLoss`.

        Arguments:
          y_true: A tensor of size [batch, num_anchors, num_classes]
          y_pred: A tensor of size [batch, num_anchors, num_classes]

        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('difference_loss'):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)

            # loss = - (y_true - y_pred)
            loss = tf.abs(y_true - y_pred)

        return loss


def rand_initialize_weights(L_in, L_out, epsilon):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections;

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    W = epsilon * (2 * np.random.rand(L_in+1, L_out) * - 1)
    return W


def set_weights(mlp):
    assert mlp.layers[0].name == "dense_1"
    in_channel, out_channel = 2, 2
    theta1 = rand_initialize_weights(in_channel, out_channel, epsilon=1)
    print(theta1.shape)
    mlp.layers[0].set_weights([theta1[:in_channel, :], theta1[-1, :]])

    in_channel, out_channel = 2, 1
    assert mlp.layers[1].name == "output"
    theta2 = rand_initialize_weights(in_channel, out_channel, epsilon=1)
    mlp.layers[1].set_weights([theta2[:in_channel, :], theta2[-1, :]])


def reset_weights(model):
    """reset_weights
    # Argument:
        model: keras Model
    """
    for layer in model.layers:
        if len(layer.get_weights()) == 2:
            weight_initializer = layer.kernel_initializer
            bias_initializer = layer.bias_initializer
            old_weights, old_biases = layer.get_weights()
            layer.set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=old_biases.shape)])

        elif len(layer.get_weights()) == 1:
            if hasattr(layer, 'kernel_initializer'):
                weight_initializer = layer.kernel_initializer
                old_weights = layer.get_weights()[0]
                layer.set_weights(
                    [weight_initializer(shape=old_weights.shape)])
            if hasattr(layer, 'bias_initializer'):
                bias_initializer = layer.bias_initializer
                old_biases = layer.get_weights()[1]
                layer.set_weights([bias_initializer(shape=old_biases.shape)])


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
    # tf.keras.losses.Huber(), yes
    # loss = tf.keras.losses.BinaryCrossentropy()
    loss = tf.keras.losses.Huber()

    # mlp.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7)) # 1e-4
    # 0.1
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.99)
    mlp.compile(loss=loss, optimizer=optimizer)

    epochs = 10000
    history = mlp.fit(dataset, epochs=epochs, verbose=2)

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
    plt.imshow(output_map, origin="lower", cmap="rainbow")
    plt.colorbar()
    plt.title("MLP output on x-y plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(log_dir, "output_map.jpg"))
    plt.show()


if __name__ == "__main__":
    main()
    difference_loss = DifferenceLoss()
    ret = difference_loss([1], [-1])
    print(ret)

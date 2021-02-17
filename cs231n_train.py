#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-16-21 19:44
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
from datetime import datetime
import numpy as np
from xor_gate_nn.cs231n.classifiers.neural_net import TwoLayerNet
import matplotlib.pyplot as plt


def get_output_map(net, N=5):
    """
    # Arguments:
        net: cs231n.classifiers.neural_net.TwoLayerNet
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    output_map = []
    for _x in x:
        for _y in y:
            _input = np.array([[_x, _y]])
            h = net.predict(_input, num_classes=1)
            output_map.append(h)
    output_map = np.asarray(output_map).reshape((N, N))

    return output_map


def main():
    # Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    model_name = "mlp_cs231n"
    date_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    prefix = os.path.join("~", "Documents", "DeepLearningData", "xor")
    subfix = os.path.join(model_name, date_time)
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))
    ckpt_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    input_size = 2
    hidden_size = 2
    output_size = 1

    net = TwoLayerNet(input_size, hidden_size,
                      output_size, std=1e-1)
    net.params['W1'] = np.random.rand(input_size, hidden_size) * 2 - 1
    net.params['b1'] = np.random.rand(hidden_size) * 2 - 1
    net.params['W2'] = np.random.rand(hidden_size, output_size) * 2 - 1
    net.params['b2'] = np.random.rand(output_size) * 2 - 1

    epochs = 30000
    stats = net.train(X, y, X, y,
                      learning_rate=0.5, learning_rate_decay=1, reg=0,
                      num_iters=epochs, batch_size=4, verbose=True)

    print('Final training loss: ', stats['loss_history'][-1])

    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.title('Training Loss history')
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.grid()
    plt.savefig(os.path.join(log_dir, "loss_history.jpg"))
    plt.show()

    output_map = get_output_map(net, N=20)
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

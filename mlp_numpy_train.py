#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-28-20 23:10
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://www.cnblogs.com/belter/p/6711160.html

import os
import numpy as np
from datetime import datetime
from xor_gate_nn.datasets.keras_fn.datasets import XOR_Dataset
from xor_gate_nn.models.keras_fn.mlp import MLP
import matplotlib.pyplot as plt

"""Neural Network for XOR
"""


def rand_initialize_weights(L_in, L_out, epsilon):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections;

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    epsilon_init = epsilon
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_gradient(z):
    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g


def forward(theta1, theta2, X):
    a_1 = np.vstack((np.array([[1]]), X.T))  # 列向量, 3*1
    z_2 = np.dot(theta1, a_1)  # 2*1
    a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))  # 3*1
    z_3 = np.dot(theta2, a_2)  # 1*1
    a_3 = sigmoid(z_3)
    h = a_3  # 预测值h就等于a_3, 1*1

    return h


def nn_cost_function(theta1, theta2, X, y):
    m = X.shape[0]  # batch_size, m=4

    # 计算所有参数的偏导数（梯度）
    D_1 = np.zeros(theta1.shape)  # Δ_1
    D_2 = np.zeros(theta2.shape)  # Δ_2
    h_total = np.zeros((m, 1))  # 所有样本的预测值, m*1, probability

    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))  # 列向量, 3*1
        z_2 = np.dot(theta1, a_1)  # 2*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))  # 3*1
        z_3 = np.dot(theta2, a_2)  # 1*1
        a_3 = sigmoid(z_3)
        h = a_3  # 预测值h就等于a_3, 1*1

        h_total[t, 0] = h

        # softmax loss gradient 最后一层每一个单元的误差, δ_3, 1*1
        delta_3 = h - y[t:t + 1, :].T
        # 第二层每一个单元的误差（不包括偏置单元）, δ_2, 2*1
        delta_2 = np.multiply(
            np.dot(theta2[:, 1:].T, delta_3), sigmoid_gradient(z_2))

        D_2 += np.dot(delta_3, a_2.T)  # 第二层所有参数的误差, 1*3
        D_1 += np.dot(delta_2, a_1.T)  # 第一层所有参数的误差, 2*3

    theta1_grad = (1.0 / m) * D_1  # 第一层参数的偏导数，取所有样本中参数，没有加正则项
    theta2_grad = (1.0 / m) * D_2

    # mean cross entropy, binary class mode
    J = (1.0 / m) * np.sum(-y * np.log(h_total) -
                           (1 - y) * np.log(1 - h_total))

    return {'theta1_grad': theta1_grad,
            'theta2_grad': theta2_grad,
            'J': J, 'h': h_total}


def get_output_map(theta1, theta2, N=5):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    output_map = []
    for _x in x:
        for _y in y:
            _input = np.array([[_x, _y]])
            h = forward(theta1, theta2, _input)
            h = h[0][0]
            output_map.append(h)
    output_map = np.asarray(output_map).reshape((N, N))

    return output_map


def train(theta1, theta2, X, y):
    iterations = 20000  # 之前的问题之二，迭代次数太少
    alpha = 0.5  # 之前的问题之三，学习率太小
    results = {'J': [], 'h': []}
    theta_history = {}

    for i in range(iterations):
        cost_fun_results = nn_cost_function(
            theta1=theta1, theta2=theta2, X=X, y=y)
        theta1_grad = cost_fun_results.get('theta1_grad')
        theta2_grad = cost_fun_results.get('theta2_grad')
        J = cost_fun_results.get('J')
        h_current = cost_fun_results.get('h')

        # Learning
        theta1 -= alpha * theta1_grad
        theta2 -= alpha * theta2_grad

        results['J'].append(J)
        results['h'].append(h_current)
        # print(i, J, h_current)
        if i == 0 or i == (iterations-1):
            print('theta1', theta1)
            print('theta2', theta2)
            theta_history['theta1_'+str(i)] = theta1.copy()
            theta_history['theta2_'+str(i)] = theta2.copy()

        if i % 10000 == 0 and i / 10000 > 0:
            plt.plot(results.get('J'))
            plt.show()

    return theta_history, results


def main():
    model_name = "mlp_numpy"
    date_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    prefix = os.path.join("~", "Documents", "DeepLearningData", "xor")
    subfix = os.path.join(model_name, date_time)
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))
    ckpt_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # XOR dataset
    # dataset = XOR_Dataset()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # 之前的问题之一，epsilon的值设置的太小
    HIDDEN_LAYER_SIZE = 2
    INPUT_SIZE = 2  # input feature
    NUM_LABELS = 1  # output class number
    theta1 = rand_initialize_weights(INPUT_SIZE, HIDDEN_LAYER_SIZE, epsilon=1)
    theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS, epsilon=1)

    theta_history, results = train(theta1, theta2, X, y)

    plt.plot(results.get('J'))
    plt.title("MLP J (softmax loss)")
    plt.xlabel("epoch")
    plt.ylabel("J")
    plt.grid()
    plt.savefig(os.path.join(log_dir, "J.jpg"))
    plt.show()

    print(theta_history)
    print(results.get('h')[0], results.get('h')[-1])

    # theta1 = np.array([[-10.15648968,   6.58788838,   6.58789596],
    #                    [3.13531709,  -7.49138826,  -7.4914178]])
    # theta2 = np.array([[7.34680943, -14.96458845, -14.95352376]])

    # print(forward(theta1, theta2, np.expand_dims(X[0], 0)))
    # print(forward(theta1, theta2, np.expand_dims(X[1], 0)))
    # print(forward(theta1, theta2, np.expand_dims(X[2], 0)))
    # print(forward(theta1, theta2, np.expand_dims(X[3], 0)))

    output_map = get_output_map(theta1, theta2, N=20)

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

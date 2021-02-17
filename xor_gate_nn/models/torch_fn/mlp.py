#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-17-21 19:10
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """MLP with BatchNorm, ReLU and Dropout
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)

        return h


def main():
    import numpy as np
    mlp = MLP()
    t = np.array([[1.0, 0.0]], dtype=np.float32)
    t = torch.from_numpy(t)
    ret = mlp(t)
    print(ret)


if __name__ == "__main__":
    main()

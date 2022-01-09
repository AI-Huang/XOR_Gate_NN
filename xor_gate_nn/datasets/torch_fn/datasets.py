#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-17-21 19:39
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import numpy as np
import tensorflow as tf

# TODO


class XOR_Dataset(TODO):
    """XOR_Dataset."""

    def __init__(
        self,
        batch_size=1,
        shuffle=False,
        seed=42,
    ):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        assert batch_size <= 4
        self.batch_size = batch_size  # one by one learning
        self.index = self._set_index_array()
        self.shuffle = shuffle

    def __getitem__(self, batch_index):
        """Gets batch at batch_index `batch_index`.

        Arguments:
            batch_index: batch_index of the batch in the Sequence.

        Returns:
            batch_x, batch_y: a batch of sequence data.
        """
        batch_size = self.batch_size

        sample_index = \
            self.index[batch_index * batch_size:(batch_index+1) * batch_size]

        batch_x = np.empty((batch_size, 2))
        batch_y = np.empty(batch_size)

        for _, i in enumerate(sample_index):
            batch_x[_, ] = self.X[i, :]
            batch_y[_] = self.y[i, :]

        return batch_x, batch_y

    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.index.shape[0] / self.batch_size))

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def _set_index_array(self):
        """_set_index_array
        """
        N = 4
        return np.arange(0, N)


def main():
    pass


if __name__ == "__main__":
    main()

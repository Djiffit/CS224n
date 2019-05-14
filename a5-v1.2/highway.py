#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np

### YOUR CODE HERE for part 1h

class Highway(nn.Module):

    def __init__(self, embedding_dim: int, dropout_rate=0.5):
        """
        :param embedding_dim: Dimension for embedding_dim
        droput: Dropout rate for dropout layer
        """
        super(Highway, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        Take minibatch of convolution
        :param x: list of length embedding_dim
        :return: highway end result, torch.Tensor of size (embedding_dim, )
        """

        x_proj = torch.relu_(self.linear1(x))
        x_gate = torch.sigmoid(self.linear2(x))

        print('-' * 50, 'weights')
        print(self.linear1.weight, self.linear1.bias)
        print(self.linear2.weight, self.linear2.bias)


        np.testing.assert_array_equal((self.embedding_dim, ), x_proj.shape)
        np.testing.assert_array_equal((self.embedding_dim, ), x_gate.shape)

        x_highway = x_gate * x_proj + (1 - x_gate) * x

        np.testing.assert_array_equal((self.embedding_dim, ), x_highway.shape)

        return self.dropout(x_highway)


### END YOUR CODE


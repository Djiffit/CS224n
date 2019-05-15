#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):

    def __init__(self, embed_size, filters, max_word_length, kernel_size =5 ) :
        """

        Initialize this stuff ::)
        :param embed_size: input_dimensions for the CNN
        :param filters: output dimensions for the CNN
        :param max_word_length: Max word length
        :param kernel_size: size of kernel for convolution
        """

        super(CNN, self).__init__()

        self.embed_size = embed_size
        self.filters = filters
        self.max_word_length = max_word_length

        self.conv1 = nn.Conv1d(embed_size, filters, kernel_size=kernel_size, bias=False)
        self.max_pool = nn.MaxPool1d(max_word_length - kernel_size + 1)



    def forward(self, input_x):
        """
        Convolutional mapping of input_x of shape input_dim onto output_dim
        :param input_x: vector of shape (batch_size, char_embed_size, max_word_length)
        :return: return Tensor of shape (batch_size, word_embed_size)
        """

        conv_x = torch.relu_(self.conv1(input_x))
        conv_out = self.max_pool(conv_x).squeeze()

        return conv_out


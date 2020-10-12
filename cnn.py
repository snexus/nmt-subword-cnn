#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, word_embed_size, char_embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.conv = nn.Conv1d(in_channels=self.char_embed_size, out_channels=self.word_embed_size,
                              kernel_size=self.kernel_size)

    def forward(self, x_reshaped):
        """
        Forward propagation for CNN class.

        Parameters
        ----------
        x_reshaped - torch tensor with shape (batch, e_char, m_word)

        Returns
        -------
        xconv_out - torch tensor with shape (batch, word_embedding)
        """
        # print("CNN: Shape of input: ", x_reshaped.shape)
        xconv = self.conv(x_reshaped)
        # print(f"CNN: Shape after convolution of size {self.kernel_size}: ", xconv.shape)
        xconv = F.relu(xconv)
        xconv_out = F.max_pool1d(xconv, kernel_size=xconv.shape[2]).squeeze(2)
        # print(f"CNN: Shape of xconv_out: ", xconv_out.shape)
        # print(f"CNN: Shape after squeezed maxpool: ", xconv_out.shape)
        return xconv_out


def test_case1():
    print("Test 1: checking shapes.")

    word_embed_size = 3
    cnn = CNN(char_embed_size=2, word_embed_size=word_embed_size, kernel_size=5)
    input = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]])
    batch_size = input.shape[0]
    input_t = torch.from_numpy(input).float()
    print(input_t)

    out = cnn(input_t)
    #
    # print("Input shape: ", input_t.shape)
    # print("Output shape: ", out.shape)
    print(out.shape)
    assert out.shape[0] == batch_size
    assert out.shape[1] == word_embed_size
    #
    print("TETS 1 PASSED.")


def test_case2():
    print("Test 2: checking shapes for multiple batches")

    word_embed_size = 3
    cnn = CNN(char_embed_size=2, word_embed_size=word_embed_size, kernel_size=5)
    input = np.array([
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    ])
    batch_size = input.shape[0]
    input_t = torch.from_numpy(input).float()
    print(input_t)

    out = cnn(input_t)
    #
    # print("Input shape: ", input_t.shape)
    # print("Output shape: ", out.shape)
    print(out.shape)
    assert out.shape[0] == batch_size
    assert out.shape[1] == word_embed_size
    #
    print("TETS 1 PASSED.")


if __name__ == '__main__':
    test_case1()
    test_case2()

### END YOUR CODE

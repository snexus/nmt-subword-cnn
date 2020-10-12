#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        """
        Highway class (direct connection)

        Parameters
        ----------
        word_embed_size (int) - size of the word embedding, should be the same size as output
                                from convolutional pooling layer.
        dropout_rate (float) - probablity of drop-out, for regularization.
        """

        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size

        self.projection = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)
        self.gate = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)


    def forward(self, xconv_out):
        """
        Forward propagation for highway class.

        Parameters
        ----------
        xconv_out - torch tensor with shape (batch, word_embedding)

        Returns
        -------
        xword_embed - torch tensor with shape (batch, word_embedding)
        """

        x_proj = F.relu(self.projection(xconv_out))
        #print("x_proj shape: ", x_proj.shape)
        x_gate = torch.sigmoid(self.gate(xconv_out))
        #print("x_gate shape: ", x_gate.shape)

        x_highway = x_gate * x_proj + (1.0 - x_gate) * xconv_out

        return x_highway

### END YOUR CODE


def test_case1():
    print("Test 1: checking shapes.")
    hw = Highway(word_embed_size=4)
    input = np.array([[1,2,3,4]])
    input_t = torch.from_numpy(input).float()

    out = hw(input_t)

    print("Input shape: ", input_t.shape)
    print("Output shape: ", out.shape)
    assert out.shape == input_t.shape

    print("TETS 1 PASSED.")

def test_case2():
    print("Test 2: checking batched shapes.")
    hw = Highway(word_embed_size=4)
    input = np.array([[1,1,1,1], [2,2,2,2]])
    input_t = torch.from_numpy(input).float()

    out = hw(input_t)

    print("Input shape: ", input_t.shape)
    print("Output shape: ", out.shape)
    assert out.shape == input_t.shape

    print("TETS 2 PASSED.")

if __name__ == '__main__':
    test_case1()
    test_case2()
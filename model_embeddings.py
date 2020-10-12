#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab, dropout_proba = 0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.vocab = vocab
        self.embed_size = embed_size
        self.pad_token_idx = vocab['<pad>']
        self.dropout = nn.Dropout(p=dropout_proba)
        self.char_embed_size = 50
        self.word_embed_size = embed_size
        self.embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=self.pad_token_idx)
        self.cnn = CNN(word_embed_size=self.word_embed_size, char_embed_size=self.char_embed_size)
        self.highway = Highway(word_embed_size=self.word_embed_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        # out = []
        # for sentence in input: # iterate over sentences
        #     # print("Dimension of sentence: ", sentence.shape)
        #
        #     emb = self.embedding(sentence).permute(0,2,1) #  (batch_size, max_word_length, embed_size=echar) -> (batch_size, embed_size=echar, max_word_length)
        #     # print("Dimension of embedding: ", emb.shape)
        #     x_conv_out = self.cnn(emb)
        #     # print("Dimension after convolution: ", x_conv_out.shape)
        #     x_highway = self.highway(x_conv_out)
        #     # print("Dimension after highway: ", x_highway.shape)
        #     xword_embed = self.dropout(x_highway) # shape (batch_size, embed_size)
        #     out.append(xword_embed)
        #
        # out = torch.stack(out, dim=0)
        # print("Dimension of out: ", out.shape)

        out = torch.zeros(input.shape[0], input.shape[1], self.word_embed_size )
        for i, sentence in enumerate(input):  # iterate over sentences
            # print("Dimension of sentence: ", sentence.shape)

            emb = self.embedding(sentence).permute(0, 2,
                                                   1)  # (batch_size, max_word_length, embed_size=echar) -> (batch_size, embed_size=echar, max_word_length)
            # print("Dimension of embedding: ", emb.shape)
            x_conv_out = self.cnn(emb)
            # print("Dimension after convolution: ", x_conv_out.shape)
            x_highway = self.highway(x_conv_out)
            # print("Dimension after highway: ", x_highway.shape)
            xword_embed = self.dropout(x_highway)  # shape (batch_size, embed_size)
            out[i] = xword_embed
            #out.append(xword_embed)

        #out = torch.stack(out, dim=0)


        return out

        ### END YOUR CODE


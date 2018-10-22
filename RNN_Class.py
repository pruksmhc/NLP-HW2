# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np 

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size, weight):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding.from_pretrained(weight)
        #self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # the input vector is size emb_size
        # I found this from Fcaebook AI's implementation, where they used an intermediate layer size of 512
        # nodes. 
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size*4, 512),
            nn.Linear(512, num_classes),
        )

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(2, batch_size, self.hidden_size)

        return hidden

    def forward(self, first_sentence_batch, second_sentence_batch, length1, length2, order_1, order_2):
        # reset hidden state

        batch_size, seq_len_one= first_sentence_batch.size()
        _, seq_len_sec = first_sentence_batch.size()

        self.hidden = self.init_hidden(batch_size)
        # get embedding of characters
        #ENCODER 
        embed1 = self.embedding(first_sentence_batch)
        embed2 = self.embedding(second_sentence_batch)
        # pack padded sequence (which means that, given a sequence that is padded, remove all the 0s so you only get)
        # an array of sum(length), word_embedding_Size (also flatten)
        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, length1.numpy(), batch_first=True)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, length2.numpy(), batch_first=True)
        # fprop though RNN
        _, self.hidden_1 = self.gru(embed1, self.hidden)
        _, self.hidden_2 = self.gru(embed2, self.hidden)
        # concat the first column 
        hidden_concat_first = torch.cat((self.hidden_1[0], self.hidden_1[1]), dim=1)
        hidden_concat_sec = torch.cat((self.hidden_2[0], self.hidden_2[1]), dim=1)

        # now, we rearrange to get the correct ordering
        hidden_concat_first = torch.index_select(hidden_concat_first, 0, order_1)
        hidden_concat_sec2 = torch.index_select(hidden_concat_sec, 0, order_2)
        final_hidden = torch.cat((hidden_concat_first, hidden_concat_sec), dim=1)
        # revrese is in the first dimension
        # undo packing, this basically inserts back the zeroes
        # we don't use this?
        # DECODER
        # sum hidden activations of RNN across time
         # DECODER
        logits = self.decoder(final_hidden)
        # so this should be a 1x num_classes vector that then needs to be soft-maxed. 
        # output = self.softmax(logits)
        return logits


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for sentence1, sentence2, length1, length2, order_1, order_2, labels in loader:
        outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
        outputs = F.softmax(outputs, dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


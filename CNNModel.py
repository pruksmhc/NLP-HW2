

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class CNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):

        super(CNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
    
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.linear1 = nn.Linear(hidden_size, num_classes*4)
        self.linear2 =  nn.Linear(num_classes*4, num_classes)

    def forward(self, first_sentence_batch, second_sentence_batch, length1, length2, order_1, order_2):
        # for this do you also feed in and encode the representations separately?
        batch_size, seq_len1 = first_sentence_batch.size()
        batch_size, seq_len2 = first_sentence_batch.size()
        embed1 = self.embedding(first_sentence_batch)
        embed2 =  self.embedding(second_sentence_batch)

        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, length1.numpy(), batch_first=True)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, length2.numpy(), batch_first=True)
        # embed dimension size torch.Size([BATCH SIZE, MAX_LEN, WORD_EMBED_SIZE])
        hidden1 = self.conv1(embed1.transpose(1,2)).transpose(1,2)
        # SAME 
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, seq_len1, hidden1.size(-1))
        #SAME
        hidden1 = self.conv2(hidden1.transpose(1,2)).transpose(1,2)
        #SAME
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, seq_len1, hidden1.size(-1))
        #SAME
        hidden2 = self.conv1(embed2.transpose(1,2)).transpose(1,2)
        #SAME
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, seq_len2, hidden2.size(-1))
        #SAME
        hidden2 = self.conv2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, seq_len2, hidden2.size(-1))

        hidden_concat_first = torch.cat((self.hidden_1[0], self.hidden_1[1]), dim=1)
        hidden_concat_sec = torch.cat((self.hidden_2[0], self.hidden_2[1]), dim=1)


        hidden = torch.sum(hidden, dim=1)
        logits = self.linear(hidden)
        return logits







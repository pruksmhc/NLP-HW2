# now, we tokenize our current dataset
import pickle
import pdb 
import numpy as np
import pandas as pd
import pprint
import numpy as np
from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder


class CNN(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, num_classes, kernel_size, weight, maxlength1, maxlength2):

        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding.from_pretrained(weight)
        self.conv1 = torch.nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.padding = 1
        self.kernel_size = kernel_size
        new_length1 = maxlength1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length1 += 1
        new_length1 = new_length1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length1 += 1
        new_length2 = maxlength2 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length2 += 1
        new_length2 = new_length2  + (2*self.padding) - (self.kernel_size -1) - 1
        new_length2 += 1
        self.maxpool1 = torch.nn.MaxPool1d(new_length1, stride=1)
        self.maxpool2 = torch.nn.MaxPool1d(new_length2, stride=1)
        self.linear1 = torch.nn.Linear(2*hidden_size, 512)
        self.linear2 =  torch.nn.Linear(512, num_classes)

    def forward(self, first_sentence_batch, second_sentence_batch, length1, length2, order_1, order_2):
        # for this do you also feed in and encode the representations separately?
        batch_size, seq_len1 = first_sentence_batch.size()
        batch_size, seq_len2 = second_sentence_batch.size()
        embed1 = self.embedding(first_sentence_batch)
        embed2 =  self.embedding(second_sentence_batch)

        # embed dimension size torch.Size([BATCH SIZE, MAX_LEN, WORD_EMBED_SIZE])
        hidden1 = self.conv1(embed1.transpose(1,2)).transpose(1,2)
        # SAME 
        new_length = seq_len1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, new_length, hidden1.size(-1))
        #SAME
        new_length = new_length + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden1 = self.conv2(hidden1.transpose(1,2)).transpose(1,2)

        #SAMEc
        # .view(batch_size, len(hidden1[]) hidden1.size(-1))
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, new_length, hidden1.size(-1))
        # Doe sit do maxpooling1D over a batch as if it was doing it fofr each row independently. 
        hidden1 = self.maxpool1(hidden1.transpose(1,2)).transpose(1,2) # we transpose the 2 dimensions
        
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, hidden1.size(-1))

        hidden2 = self.conv1(embed2.transpose(1,2)).transpose(1,2)
        #SAME
        new_length = seq_len2 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, new_length, hidden2.size(-1))
        #SAME
        new_length = new_length + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden2 = self.conv2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, new_length, hidden2.size(-1))

        hidden2 = self.maxpool2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, hidden2.size(-1))

        # order and concat
        hidden_concat_first = torch.index_select(hidden1, 0, order_1)
        hidden_concat_sec = torch.index_select(hidden2, 0, order_2)
        final_hidden = torch.cat((hidden_concat_first, hidden_concat_sec), dim=1)
        # DECODE
        logits = self.linear1(final_hidden)
        logits = F.leaky_relu(logits)
        logits = self.linear2(logits)
        return logits


class CNNWithDropout(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, num_classes, kernel_size, weight, maxlength1, maxlength2, dw):

        super(CNNWithDropout, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding.from_pretrained(weight)
        self.conv1 = torch.nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.padding = 1
        self.kernel_size = kernel_size
        new_length1 = maxlength1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length1 += 1
        new_length1 = new_length1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length1 += 1
        new_length2 = maxlength2 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length2 += 1
        new_length2 = new_length2  + (2*self.padding) - (self.kernel_size -1) - 1
        new_length2 += 1
        self.dropout = torch.nn.Dropout(p=dw)
        self.maxpool1 = torch.nn.MaxPool1d(new_length1, stride=1)
        self.maxpool2 = torch.nn.MaxPool1d(new_length2, stride=1)
        self.linear1 = torch.nn.Linear(2*hidden_size, 512)
        self.linear2 =  torch.nn.Linear(512, num_classes)

    def forward(self, first_sentence_batch, second_sentence_batch, length1, length2, order_1, order_2):
        # for this do you also feed in and encode the representations separately?
        batch_size, seq_len1 = first_sentence_batch.size()
        batch_size, seq_len2 = second_sentence_batch.size()
        embed1 = self.embedding(first_sentence_batch)
        embed2 =  self.embedding(second_sentence_batch)

        # embed dimension size torch.Size([BATCH SIZE, MAX_LEN, WORD_EMBED_SIZE])
        hidden1 = self.conv1(embed1.transpose(1,2)).transpose(1,2)
        # SAME 
        new_length = seq_len1 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, new_length, hidden1.size(-1))
        hidden1 = self.dropout(hidden1)
        #SAME
        new_length = new_length + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden1 = self.conv2(hidden1.transpose(1,2)).transpose(1,2)
        #SAMEc
        # .view(batch_size, len(hidden1[]) hidden1.size(-1))
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, new_length, hidden1.size(-1))
        hidden1 = self.dropout(hidden1)
        # Doe sit do maxpooling1D over a batch as if it was doing it fofr each row independently. 
        hidden1 = self.maxpool1(hidden1.transpose(1,2)).transpose(1,2) # we transpose the 2 dimensions
        
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size, hidden1.size(-1))
        hidden1 = self.dropout(hidden1)
        hidden2 = self.conv1(embed2.transpose(1,2)).transpose(1,2)
        #SAME
        new_length = seq_len2 + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, new_length, hidden2.size(-1))
        #SAME
        hidden2 = self.dropout(hidden2)
        new_length = new_length + (2*self.padding) - (self.kernel_size -1) - 1
        new_length += 1
        hidden2 = self.conv2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, new_length, hidden2.size(-1))
        hidden2 = self.dropout(hidden2)
        hidden2 = self.maxpool2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size, hidden2.size(-1))
        # order and concat
        hidden_concat_first = torch.index_select(hidden1, 0, order_1)
        hidden_concat_sec = torch.index_select(hidden2, 0, order_2)
        final_hidden = torch.cat((hidden_concat_first, hidden_concat_sec), dim=1)

        # DECODE
        logits = self.linear1(final_hidden)
        logits = F.leaky_relu(logits)
        logits = self.linear2(logits)
        return logits


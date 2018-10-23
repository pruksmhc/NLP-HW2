
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
from sklearn.preprocessing import OneHotEncoder
from EntailmentDataLoader import *
from HyperSearch import * 

current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
#train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
current_matrix = pickle.load(open("idx2vectorfasttext50K", "rb"))
weights = pickle.load(open("weights.pkl", "rb"))
train_text_tokenized = pd.read_pickle("train_token_indexed.pkl")
val_text_tokenized = pd.read_pickle("val_token_indexed.pkl")
BATCH_SIZE = 32
# should get 4 epochs
train_dataset = NewsGroupDataset(train_text_tokenized["sentence1"].values.tolist(), train_text_tokenized["sentence2"].values.tolist(), train_text_tokenized["label"].values.tolist())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=entailment_collate_func_concat,
                                           shuffle=True)

val_dataset = NewsGroupDataset(val_text_tokenized["sentence1"].values.tolist(), val_text_tokenized["sentence2"].values.tolist(), val_text_tokenized["label"].values.tolist())

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=entailment_collate_func_concat,
                                           shuffle=True)

weights = pickle.load(open("weights.pkl", "rb"))
dirlink = "results/"
num_epochs = 5
"""
Train for the CNN, saving the best values. 
"""
lr = 0.001
searcher = HyperSearch(dirlink, num_epochs, lr)
# Hidden sizes
rnn_default_vals = {"emb_size":300, "num_layers":1, "num_classes":3, "weight":torch.FloatTensor(weights)}
cnn_default_vals = {"emb_size":300, "hidden_size":600, "kernel_size":3, "num_classes":3, "weight":torch.FloatTensor(weights), "maxlength1":50, "maxlength2":28}
searcher.search_parameters("hidden_size","RNN", [50, 100, 500, 1000], rnn_default_vals, train_loader, val_loader, "Accuracy Training/Validation Curve for RNN with Hidden Size = ")
default_vals = {"emb_size":300, "num_layers":1, "num_classes":3, "weight":torch.FloatTensor(weights)}
searcher.search_parameters("hidden_size","CNN", [50, 100, 500, 1000], cnn_default_vals, train_loader, val_loader, "Accuracy Training/Validation Curve for CNN with Hidden Size = ")
cnn_default_vals = {"emb_size":300, "hidden_size":600, "hidden_size": 500, "num_classes":3, "weight":torch.FloatTensor(weights), "maxlength1":50, "maxlength2":28}
searcher.search_parameters("kernel_size","CNN", [1, 3, 10, 20, 25], default_vals, train_loader, val_loader, "Accuracy Training/Validation Curve for Hidden Size = ")
#rnn_default_vals = {"emb_size":300, "num_layers":1, "num_classes":3, "hidden_size": 500,"weight":torch.FloatTensor(weights)}
#searcher.search_parameters("hidden_size","RNN", [50, 100, 500, 1000], rnn_default_vals, train_loader, val_loader, "Accuracy Training/Validation Curve for RNN with Hidden Size = ")
#rnn_default_vals = {"dropout:"}
# blah blah you can add dorpout just normally. 




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
from CNNModel import *


MAX_SENTENCE_LENGTH_FIRST = 50 
MAX_SENTENCE_LENGTH_SECOND = 28 
EMBED_DIM = 300


class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list_first_sentence, data_list_second_sentence, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 
        Inspired by https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649

        """
        self.datasets = [data_list_first_sentence, data_list_second_sentence]
        self.target_list = target_list
        lengths1 = [len(x) for x in data_list_first_sentence]
        self.max_length1 = max(lengths1)
        lengths2 = [len(x) for x in data_list_second_sentence]
        self.max_length2 = max(lengths2)
        assert (len(self.datasets[0]) == len(self.target_list)) and (len(self.datasets[1]) == len(self.target_list)) 

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, key):
        sentences = tuple(d[key] for d in self.datasets)
        lengths = tuple(len(d[key]) for d in self.datasets)
        label = self.target_list[key]
        return [sentences, lengths, label, self.max_length1, self.max_length2]

def get_order(sorted_list, to_construct):
    order = []
    for elt in to_construct:
        index = []
        for i in range(len(sorted_list)):
            s_elt = sorted_list[i]
            if s_elt == elt:
                index = i
        order.append(index)
    return order

def entailment_collate_func_concat(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    first_data_list = []
    second_data_list = []
    label_list = []
    length_list_first = []
    length_list_second = []
    data_list_first = []
    data_list_second = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    #pdb.set_trace()
    for datum in batch:
        first_data_list.append(datum[0][0])
        second_data_list.append(datum[0][1])
        length_list_first.append(datum[1][0])
        length_list_second.append(datum[1][1])
        label_list.append(datum[2])
    sorted_first = sorted(first_data_list, key=lambda e: len(e), reverse=True)
    # this is the sorted data list. 
    sorted_second = sorted(second_data_list, key=lambda e: len(e), reverse=True)
    order_one_to_pass =  get_order(sorted_first, first_data_list)
    order_two_to_pass =  get_order(sorted_second, second_data_list)

    order_one = sorted(range(len(length_list_first)), key=lambda k: len(first_data_list[k]), reverse=True)
    order_two = sorted(range(len(length_list_second)), key=lambda k: len(second_data_list[k]), reverse=True)
    length_first =  sorted(length_list_first, reverse=True)
    length_second = sorted(length_list_second, reverse=True)

    # Asser tthat the indexing is the same 
    #pdb.set_trace()
    for i in range(len(sorted_first)):
        elt = sorted_first[i]
        assert (np.array(elt)!=0).sum() == (np.array(first_data_list[order_one[i]])!=0).sum()
        elt = sorted_second[i]
        assert (np.array(elt)!=0).sum() == (np.array(second_data_list[order_two[i]])!=0).sum()
    # padding

    for i in range(len(length_list_first)):
        elt = first_data_list[i]
        assert (np.array(elt)!=0).sum() == (np.array(sorted_first[order_one_to_pass[i]])!=0).sum()
        elt = second_data_list[i]
        assert (np.array(elt)!=0).sum() == (np.array(sorted_second[order_two_to_pass[i]])!=0).sum()

    for i in range(len(batch)):
        # Do e first do this and then this? 
        first_sentence = sorted_first[i]
        second_sentence = sorted_second[i]
        first_sentence.extend([0]*(batch[0][3]- len(first_sentence)))
        second_sentence.extend([0]*(batch[0][4]-len(second_sentence)))
        data_list_first.append(first_sentence)
        data_list_second.append(second_sentence)
    return [torch.LongTensor(data_list_first), torch.LongTensor(data_list_second), torch.LongTensor(length_first),  torch.LongTensor(length_second), torch.LongTensor( order_one_to_pass), torch.LongTensor( order_two_to_pass), torch.LongTensor(label_list)]


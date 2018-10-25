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
from RNN_Class import *


MAX_SENTENCE_LENGTH_FIRST = 50 
MAX_SENTENCE_LENGTH_SECOND = 28 
EMBED_DIM = 300
# what's the vocab size? 
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
        assert (len(self.datasets[0]) == len(self.target_list)) and (len(self.datasets[1]) == len(self.target_list)) 

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, key):
	    sentences = tuple(d[key] for d in self.datasets)
	    lengths = tuple(len(d[key]) for d in self.datasets)
	    label = self.target_list[key]
	    return [sentences,lengths, label]

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
    	first_sentence.extend([0]*(MAX_SENTENCE_LENGTH_FIRST- len(first_sentence)))
    	second_sentence.extend([0]*(MAX_SENTENCE_LENGTH_SECOND-len(second_sentence)))
    	data_list_first.append(first_sentence)
    	data_list_second.append(second_sentence)
    return [torch.LongTensor(data_list_first), torch.LongTensor(data_list_second), torch.LongTensor(length_first),  torch.LongTensor(length_second), torch.LongTensor( order_one_to_pass), torch.LongTensor( order_two_to_pass), torch.LongTensor(label_list)]


 # load word embeddings from GloVe 
def load_glove():
	glove_home = 'glove.6B/'
	words_to_load = 50000
	with open(glove_home + 'glove50d.txt') as f:
	    loaded_embeddings = np.zeros((words_to_load, 50))
	    words = {}
	    idx2words = {}
	    for i, line in enumerate(f):
	        if i >= words_to_load: 
	            break
	        s = line.split()
	        loaded_embeddings[i, :] = np.asarray(s[1:])
	        words[s[0]] = i
	        idx2words[i] = s[0]
	return idx2words, words, loaded_embeddings

def load_fasttext():
    ft_home = ''
    words_to_load = 50000
    with open(ft_home + 'wiki-news-300d-1M.vec') as f:
        loaded_embeddings_ft = np.zeros((words_to_load, 300))
        words_ft = {}
        idx2words_ft = {}
        for i, line in enumerate(f):
            if i >= words_to_load: 
                break
            s = line.split()
            loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            words_ft[s[0]] = i
            idx2words_ft[i] = s[0]
    return idx2words_ft, words_ft, loaded_embeddings_ft

def tokenize_all_vectors(tokens, words, loaded_embeddings, unknown_vector):
    current_word2idx = {}
    current_matrix = {}
    current_matrix[1] = unknown_vector
    current_matrix[0]  = [0]* EMBED_DIM # this is just the padding
    keys = np.array(list(words.keys()))
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in keys:
            current_matrix[i+2] = loaded_embeddings[words[token]]
            current_word2idx[token] = i+2
            if len(current_matrix[i+2]) == 1:
                current_matrix[i+2] = current_matrix[i+2][0]
        else:
            current_word2idx[token] = 1 # it's not in the vocabulary, so map to 1. 
    # make sure that the number of rows of current matrix is one more than the cardinality of set of all unique 
    # indices in current_word2idx (since 0 shuold not appera in the current word2idx)
    unique_embeds = set(tuple(row) for row in current_matrix.values())
    assert len(unique_embeds) == (len(set(current_word2idx.values())) + 1)
    return current_matrix, current_word2idx

def tokenize_labels(data):
	labels = {"neutral":0, "contradiction": 1, "entailment": 2}
	data["label"] = data["label"].apply(lambda i: labels[i])
	return data

def tokenize_on_glove_vectors(text, current_word2idxe):
    result = []
    # OH MY GOD I FOUND IT. THERE IS INDEED A BUG IN HERE. 
    for word in text:
        word = word.lower()
        if word in list(current_word2idx.keys()):
            result.append(current_word2idx[word])
        else:
            #pdb.set_trace()
            result.append(1) # this is the unknown vector
    return result


def generate_indexed_val(current_word2idx):
    val_text_tokenized = pd.read_pickle("val_text_tokenized.pkl")
    val_text_tokenized["sentence1"] = val_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    val_text_tokenized["sentence2"] = val_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    val_text_tokenized = tokenize_labels(val_text_tokenized)
    pickle.dump(val_text_tokenized, open("val_token_indexed.pkl", "wb"))

def generate_indexed_train(current_word2idx):
    hey = pd.read_csv("hw2_data/snli_train.tsv", sep="\t")
    train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
    train_text_tokenized["sentence2"] = train_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    train_text_tokenized["sentence1"] = train_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    pickle.dump(train_text_tokenized, open("train_token_indexed.pkl", "wb"))

def generate_weights(column_keys, max_index):
    """
    column_keys = list(current_matrix.keys()) # curernt marix
    max_index = max(column_keys) # this goes up to a 1. 
    """
    weights = []
    for i in range(max_index+1):
        if i in column_keys:
            weights.append(current_matrix[i])
        else:
            weights.append(current_matrix[1])
    pickle.dump(weights, open("weights.pkl", "wb"))

def generate_initial_id2wordv():
    all_tokens = pickle.load(open("train_tokens.pkl", "rb")) # tHIS IS ONLY FOR GLOVE. 
    idx2words, words, loaded_embeddings = load_fasttext()
    unknown_vector = list(np.random.normal(scale=0.6, size=(EMBED_DIM, )))
    words["UNK"] = unknown_vector
    glove_vocab = np.array(list(words.keys()))
    idx2words[len(glove_vocab) - 1] = "UNK"
    # check the all_tokens is working. 
    current_matrix, current_word2idx = tokenize_all_vectors(list(all_tokens), words, loaded_embeddings, unknown_vector)
    pickle.dump(current_word2idx, open("word2idxfasttext50K", "wb"))
    pickle.dump(current_matrix, open("idx2vectorfasttext50K", "wb"))



current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
#train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
current_matrix = pickle.load(open("idx2vectorfasttext50K", "rb"))
weights = pickle.load(open("weights.pkl", "rb"))
train_text_tokenized = pd.read_pickle("train_token_indexed.pkl")
val_text_tokenized = pd.read_pickle("val_token_indexed.pkl")

BATCH_SIZE = 32

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


"""
NOW THE TRAINING 
"""
weights = pickle.load(open("weights.pkl", "rb"))
model = RNN(emb_size=300, hidden_size=600, num_layers=1, num_classes=3,  weight=torch.FloatTensor(weights))
learning_rate = 0.001
num_epochs = 100 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (sentence1, sentence2, length1, length2, order_1, order_2, labels) in enumerate(train_loader):
        # make sure that the order is the smae, 
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
        #print("Predictions at this step is")
        loss = criterion(outputs, labels)
        #print("loss is"+ str(loss.item()))
        # Backward and optimize
        loss.backward()
        optimizer.step()
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        norm_type = 2
        total_norm = 0

        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            s_outputs = F.softmax(outputs, dim=1)
            total = labels.size(0)
            predicted = s_outputs.max(1, keepdim=True)[1]
            correct = predicted.eq(labels.view_as(predicted)).sum().item()
            train_acc = (100 * correct / total)
            print("Train accuracy is" + str(train_acc))

            val_acc = test_model(val_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, total_step, val_acc))



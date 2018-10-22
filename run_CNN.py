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



train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
val_text_tokenized = pd.read_pickle("val_text_tokenized.pkl")
train_text_tokenized_mnli= pd.read_pickle("MNLI_train_text_tokenized.pkl")
val_text_tokenized_mnli = pd.read_pickle("MNLI_train_text_tokenized.pkl")
snli_data = train_text_tokenized
MAX_SENTENCE_LENGTH_FIRST =max(snli_data["sentence1"].apply(lambda x: len(x)).tolist())
MAX_SENTENCE_LENGTH_SECOND = max(snli_data["sentence2"].apply(lambda x: len(x)).tolist())
all_tokens = pickle.load(open("train_tokens.pkl", "rb"))
VOCAB_NUM = len(all_tokens)
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
    sorted_second = sorted(second_data_list, key=lambda e: len(e), reverse=True)
    order_one = sorted(range(len(length_list_first)), key=lambda k: len(first_data_list[k]), reverse=True)
    order_two = sorted(range(len(length_list_second)), key=lambda k: len(second_data_list[k]), reverse=True)
    length_first =  sorted(length_list_first, reverse=True)
    length_second = sorted(length_list_second, reverse=True)

    # Asser tthat the indexing is the same 
    #pdb.set_trace()
    for i in range(len(sorted_first)):
   		elt = sorted_first[i]
   		assert len(elt) == len(first_data_list[order_one[i]])
   		elt = sorted_second[i]
   		assert len(elt) == len(second_data_list[order_two[i]])
    # padding
    for i in range(len(batch)):
    	# Do e first do this and then this? 
    	first_sentence = sorted_first[i]
    	second_sentence = sorted_second[i]
    	first_sentence.extend([0]*(MAX_SENTENCE_LENGTH_FIRST- len(first_sentence)))
    	second_sentence.extend([0]*(MAX_SENTENCE_LENGTH_SECOND-len(second_sentence)))
    	data_list_first.append(first_sentence)
    	data_list_second.append(second_sentence)
    return [torch.LongTensor(data_list_first), torch.LongTensor(data_list_second), torch.LongTensor(length_first),  torch.LongTensor(length_second), torch.LongTensor( order_one), torch.LongTensor( order_two), torch.LongTensor(label_list)]

# create pytorch dataloader
#train_loader = NewsGroupDataset(train_data_indices, train_targets)
#val_loader = NewsGroupDataset(val_data_indices, val_targets)
#test_loader = NewsGroupDataset(test_data_indices, test_targets)

#for i, (data, lengths, labels) in enumerate(train_loader):
#    print (data)
#    print (labels)
#    break
# you want to match the index to words. 
# but also the words -> index. 
# we already have th e trained_tokens. 

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

def tokenize_all_vectors(tokens, glove_vocab, loaded_embeddings):
	current_word2idx = {}
	current_matrix = {}
	current_matrix[1] = unknown_vector # this is the unknown. 
	current_matrix[0]  = [0]* EMBED_DIM # this is just the padding
	for i in range(len(tokens)):
		token = tokens[i].lower()
		index_vocab = np.nonzero(glove_vocab == token)
		if index_vocab[0].shape[0] > 0:
			current_matrix[i+2] = loaded_embeddings[index_vocab[0]]	
			if len(current_matrix[i+2]) == 1:
				current_matrix[i+2] = current_matrix[i+2][0]
				current_word2idx[token] = i+2
		else:
			current_word2idx[token] = 1 # it's not in the vocabulary, so map to 1. 
	return current_matrix, current_word2idx

def tokenize_labels(data):
	labels = {"neutral":0, "contradiction": 1, "entailment": 2}
	data["label"] = data["label"].apply(lambda i: labels[i])
	return data

def tokenize_on_glove_vectors(text, current_word2idx, vocab_size):
    result = []
    for word in text:
        word = word.lower()
        if word in list(current_word2idx.keys()):
            result.append(current_word2idx[word])
        else:
            result.append(1) # this is the unknown vector
    return result

# get all tokens
"""
all_tokens = pickle.load(open("train_tokens.pkl", "rb"))
idx2words, words, loaded_embeddings = load_fasttext()
EMBED_DIM = 300
unknown_vector = np.random.normal(scale=0.6, size=(EMBED_DIM, ))
words["UNK"] = unknown_vector
glove_vocab = np.array(list(words.keys()))
idx2words[len(glove_vocab) - 1] = "UNK"
current_matrix, current_word2idx = tokenize_all_vectors(list(all_tokens), glove_vocab, loaded_embeddings)
pickle.dump(current_word2idx, open("word2idxfasttext50K", "wb"))
pickle.dump(current_matrix, open("idx2vectorfasttext50K", "wb"))
all_tokens = pickle.load(open("train_tokens.pkl", "rb"))
"""

current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
#train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
current_matrix = pickle.load(open("idx2vectorfasttext50K", "rb"))

vocab_size = len(current_word2idx) # this is the vocab size
#train_text_tokenized = train_text_tokenized.iloc[:20000] # just do a batch of 20 for testing. 
#train_text_tokenized["sentence1"] = train_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx, vocab_size))

#train_text_tokenized["sentence2"] = train_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx, vocab_size))
# sort the sentneces based on the length of the two sentences combined. 
#pickle.dump(train_text_tokenized, open("train_token_indexed.pkl", "wb"))

# we need to sort by decreasing order first. 
train_text_tokenized = pd.read_pickle("train_token_indexed.pkl")
val_text_tokenized = pd.read_pickle("val_token_indexed.pkl")
BATCH_SIZE = 32

train_dataset = NewsGroupDataset(train_text_tokenized["sentence1"].values.tolist(), train_text_tokenized["sentence2"].values.tolist(), train_text_tokenized["label"].values.tolist())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=entailment_collate_func_concat,
                                           shuffle=True)
#val_text_tokenized = val_text_tokenized.iloc[:2000] 

#val_text_tokenized["sentence1"] = val_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx, vocab_size))

#val_text_tokenized["sentence2"] = val_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx, vocab_size))

val_dataset = NewsGroupDataset(val_text_tokenized["sentence1"].values.tolist(), val_text_tokenized["sentence2"].values.tolist(), val_text_tokenized["label"].values.tolist())

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=entailment_collate_func_concat,
                                           shuffle=True)
# sort the sentneces based on the length of the two sentences combined. 



"""
NOW THE TRAINING 
"""
column_keys = list(current_matrix.keys())
max_index = max(column_keys)
weights = []
for i in range(max_index+1):
	if i in column_keys:
		# if index is in the curent matrix, thus it's an index thatw will be accessed
		weights.append(current_matrix[i])
	else:
		weights.append(current_matrix[1]) # else, there's no glove vector (it shouldn't access anyways due to 
		# how we tokenized the vectors and built current_matrix at the same time. )
model = CNN(emb_size=300, hidden_size=600,  num_classes=3, vocab_size=len(current_word2idx), kernel_size =3, weight=torch.FloatTensor(weights))
learning_rate = 0.1
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    print(epoch)
    for i, (sentence1, sentence2, length1, length2, order_1, order_2, labels) in enumerate(train_loader):
        # make sure that the order is the smae, 
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            train_acc = test_model(train_loader, model)
            print("Train acc")
            print(str(train_acc))
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, total_step, val_acc))

# they hsouldall be of the right siz.e 
# then you have the data_loader. 
#train_iter = data.BucketIterator( 
#	dataset=mt_train, batch_size=32,  


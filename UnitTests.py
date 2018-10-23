
import pandas as pd
import pdb
import pickle
# Unit testing 

def check_test_encoding():
	# Chekc htat the test encoding is equal to the tokenizatio 
	# Check tha tth elable is correct for each one. 

	gold_data = pd.read_csv("hw2_data/snli_train.tsv",  sep='\t', names=["sentence1", "sentence2", "label"])
	train_tokenized = pd.read_pickle("train_token_indexed.pkl")
	current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
	current_idx2word = {value: key for key, value in current_word2idx.items()}
	pdb.set_trace()
	current_matrix = pickle.load(open("idx2vectorfasttext50K", "rb"))
	labels_dict = {0:"neutral", 1:"contradiction", 2:"entailment"}
	# so we should have it such that 
	for i in range(1, len(gold_data)):
		grow = gold_data.iloc[i]
		row = train_tokenized.iloc[i-1]
		gs1 = grow["sentence1"]
		s1 = row["sentence1"]
		gs2 = grow["sentence2"]
		s2 = row["sentence2"]		
		glabel = grow["label"]
		label = row["label"]	
		assert glabel == labels_dict[label]
		for j in range(len(s1)):
			if s1[j] != 1: # if unknown, then we don't care. 
				if current_idx2word[s1[j]] not in gs1.lower():
					pdb.set_trace()
					print("there's something wrong in instance "+ str(i))
					print(current_idx2word[s1[j]])
					print(gs1)
		for j in range(len(s2)):
			if s2[j] != 1:
				if current_idx2word[s2[j]] not in gs2.lower():
					pdb.set_trace()
					print("there's something wrong in instance "+ str(i))

check_test_encoding()
import pandas as pd
import pdb 

mnli_train = pd.read_pickle("MNLI_train_text_tokenized.pkl")
mnli_val = pd.read_pickle("MNLI_val_text_tokenized.pkl")
gold_train = pd.read_csv("hw2_data/mnli_val.tsv", sep="\t")
pdb.set_trace()
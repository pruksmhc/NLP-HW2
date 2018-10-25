import pandas as pd
import pdb 
import pickle
from CNNModel import * 
from RNN_Class import *
from EntailmentDataLoader import *


current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
#train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")


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

def generate_indexed_data(genre, current_word2idx):
    train_text_tokenized = pd.read_pickle("parsedMNLIGenre/val_"+genre+".pkl")
    train_text_tokenized["sentence2"] = train_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    train_text_tokenized["sentence1"] = train_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    train_text_tokenized = tokenize_labels(train_text_tokenized)
    pickle.dump(train_text_tokenized, open("parsedMNLIGenre/val_"+genre+"_indexed.pkl", "wb"))

current_word2idx = pickle.load(open("word2idxfasttext50K", "rb"))
#train_text_tokenized = pd.read_pickle("train_text_tokenized.pkl")
current_matrix = pickle.load(open("idx2vectorfasttext50K", "rb"))
weights = pickle.load(open("weights.pkl", "rb"))

def test_for_genre(genre, model, link, CNN_or_RNN):
    if CNN_or_RNN == "CNN":
        val_text_tokenized = pd.read_pickle("parsedMNLIGenre/val_"+genre+"_indexed.pkl")
        maxlen1 = max([len(x) for x in val_text_tokenized["sentence1"].values.tolist()])
        maxlen2 = max([len(x) for x in val_text_tokenized["sentence2"].values.tolist()])   
        loaded_model = CNN(emb_size=300, hidden_size=250, kernel_size=1,num_classes=3, maxlength1=maxlen1, maxlength2=maxlen2, weight=torch.FloatTensor(weights))

    val_text_tokenized = pd.read_pickle("parsedMNLIGenre/val_"+genre+"_indexed.pkl")
    # should get 4 epochs
    pdb.set_trace()
    val_dataset = NewsGroupDataset(val_text_tokenized["sentence1"].values.tolist(), val_text_tokenized["sentence2"].values.tolist(), val_text_tokenized["label"].values.tolist())
    BATCH_SIZE = 32
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=entailment_collate_func_concat,
                                           shuffle=True)
    loaded_model.load_state_dict(torch.load(link))
    val_acc = test_model(val_loader, model)
    print("For genre" + genre)
    print(str(val_acc))


def get_examples(loader, model):
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
        predictions = predicted.eq(labels.view_as(predicted)).sum().item()
        pdb.set_trace() 


    return (100 * correct / total)



loaded_model = RNN(emb_size=300, hidden_size=250, num_layers=1, num_classes=3,  weight=torch.FloatTensor(weights))
print("FOR RNN BEST MODEL ")
#test_for_genre("fiction",loaded_model, 'results/RNN/hidden_size/250/model_states', "RNN" )

print("FOR CNN BEST MODEL")
test_for_genre("fiction",loaded_model, 'results/CNN/kernel_size/1/model_states', "CNN" )
test_for_genre("slate",loaded_model, 'results/CNN/kernel_size/1/model_states', "CNN")
test_for_genre("travel",loaded_model,'results/CNN/kernel_size/1/model_states', "CNN" )
test_for_genre("tele",loaded_model, 'results/CNN/kernel_size/1/model_states' , "CNN")
test_for_genre("gov",loaded_model, 'results/CNN/kernel_size/1/model_states', "CNN" )








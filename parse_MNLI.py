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
    train_text_tokenized = pd.read_pickle("parsedMNLIGenre/train_"+genre+".pkl")
    train_text_tokenized["sentence2"] = train_text_tokenized["sentence2"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    train_text_tokenized["sentence1"] = train_text_tokenized["sentence1"].apply(lambda x: tokenize_on_glove_vectors(x, current_word2idx))
    train_text_tokenized = tokenize_labels(train_text_tokenized)
    pickle.dump(train_text_tokenized, open("parsedMNLIGenre/train_"+genre+"_indexed.pkl", "wb"))

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


def train_further(genre, model, link):
    train_text_tokenized =  pd.read_pickle("parsedMNLIGenre/train_"+genre+"_indexed.pkl")
    val_text_tokenized = pd.read_pickle("parsedMNLIGenre/val_"+genre+"_indexed.pkl")
    # should get 4 epochs
    train_dataset = NewsGroupDataset(train_text_tokenized["sentence1"].values.tolist(), train_text_tokenized["sentence2"].values.tolist(), train_text_tokenized["label"].values.tolist())
    val_dataset = NewsGroupDataset(val_text_tokenized["sentence1"].values.tolist(), val_text_tokenized["sentence2"].values.tolist(), val_text_tokenized["label"].values.tolist())
    BATCH_SIZE = 32
    train_loader =   torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=entailment_collate_func_concat,
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=entailment_collate_func_concat,
                                            shuffle=True)
    pdb.set_trace()
    model.load_state_dict(torch.load(link))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # train 
    num_epochs = 2
    total_step = len(train_loader)
    val_accs = []
    train_accs = []
    for epoch in range(num_epochs):
        pdb.set_trace()
        v_accs = []
        t_accs = []
        for i, (sentence1, sentence2, length1, length2, order_1, order_2, labels) in enumerate(train_loader):
            print(i)
            model.train()
            optimizer.zero_grad()
            outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # we should have 10 in between. so 10 per epoch. 
            if i > 0 and i % 100 == 0:
                # there should be 2 per epoch
                s_outputs = F.softmax(outputs, dim=1)
                total = labels.size(0)
                predicted = s_outputs.max(1, keepdim=True)[1]
                correct = predicted.eq(labels.view_as(predicted)).sum().item()
                train_acc = (100 * correct / total)
                print("Train accuracy is" + str(train_acc))
                t_accs.append(train_acc)
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                           epoch+1, num_epochs, i+1, total_step, val_acc))
                v_accs.append(val_acc)
        val_accs.append(v_accs)
        train_accs.append(t_accs)
    # then,  test the model on the val_loader for mNLI with 2 epochs and lower leanring rate. 
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


# Now, calculate the numbe rof trainable parameters
#model = RNN(emb_size=300, hidden_size=50, num_layers=1, num_classes=3,  weight=torch.FloatTensor(weights))
#print("FOR RNN BEST MODEL ")
#est_for_genre("fiction",loaded_model, 'results/RNN/hidden_size/250/model_states', "RNN" )
links = [ 'results/CNN/kernel_size/1/model_states', 'results/CNN/kernel_size/5/model_states', 'results/CNN/kernel_size/10/model_states']

for link in links:
    hs = int(link.split("kernel_size/")[1].split("/")[0])
    model = CNNWithDropout(emb_size=300, hidden_size=250, kernel_size=hs, maxlength1=50, maxlength2=28, num_classes=3,  weight=torch.FloatTensor(weights), dw=0.5)
   # model = RNN(emb_size=300, hidden_size=250, num_layers=1, num_classes=3,  weight=torch.FloatTensor(weights))
    model.load_state_dict(torch.load(link))
    print(link)
    num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            size = np.prod(np.shape(param))
            num += size
    print(num)
#train_further("fiction", model, link)
train_further("slate", model, link)





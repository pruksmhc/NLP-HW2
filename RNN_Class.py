# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size, weight):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding.from_pretrained(weight)
        #self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first=True)
        # the input vector is size emb_size
        self.linear = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers, batch_size, self.hidden_size)

        return hidden

    def forward(self, first_sentence_batch, second_sentence_batch, length1, length2, order_1, order_2):
        # reset hidden state

        batch_size, seq_len = x.size()

        self.hidden = self.init_hidden(batch_size)

        # get embedding of characters
        pdb.set_trace()
        embed1 = self.embedding(first_sentence_batch)
        embed2 = self.embedding(second_sentence_batch)
        # pack padded sequence (which means that, given a sequence that is padded, remove all the 0s so you only get)
        # an array of sum(length), word_embedding_Size (also flatten)
        embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, length1.numpy(), batch_first=True)
        embed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, length2.numpy(), batch_first=True)
        # fprop though RNN
        rnn_out_1, self.hidden_1 = self.rnn(embed1, self.hidden)
        rnn_out_2, self.hidden_2 = self.rnn(embed2, self.hidden)
        # undo packing, this basically inserts back the zeroes
        rnn_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_1, batch_first=True)
        rnn_out_2, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_1, batch_first=True)
        out_final = rnn_out_1.concat(rnn_out_2)
        # sum hidden activations of RNN across time
        rnn_final = torch.sum(rnn_final, dim=1)

        logits = self.linear(rnn_final)
        # so this should be a 1x num_classes vector that then needs to be soft-maxed. 
        return logits


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, lengths_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, lengths_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


#model = RNN(emb_size=100, hidden_size=200, num_layers=2, num_classes=5, vocab_size=len(id2char))

learning_rate = 3e-4
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
train_loader = pickle.load(open( "trainloader", "rb"))
for epoch in range(num_epochs):
    for i, (sentence1, sentence2, length1, length2, order_1, order_2, labels) in enumerate(train_loader):
        import pdb; pdb.set_trace()
        # make sure that the order is the smae, 
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data, lengths)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
#import torchvison.models as models

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, batch_first=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding)

        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.relu = nn.functional.relu()

    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        
        predictions = self.out(outputs)
        # nn.functional.relu(predictions)
        
        predictions = self.softmax(predictions)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class EncodeToDecode(nn.Module):
    def __init__(self, encoder, decoder, input_size, device):
        super(EncodeToDecode, self).__init__()
        self.encoder  = encoder
        self.decoder = decoder
        self.device =  device
        self.input_size = input_size

    def forward(self, source, target):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.input_size
    
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)

        x = target[0]
        
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            x = target[t]

        return outputs
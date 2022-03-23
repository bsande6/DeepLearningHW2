from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
import json
import numpy as np
from data_loader import MLDSDataset
from seq2seq import EncodeToDecode, EncoderRNN, DecoderRNN
import math

MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
BOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkown word token 

device = torch.device('cuda:14' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = False
def train(dataloader, model, loss_fn, optimizer, num_epochs, batch_size):
    model.train() 
    for epoch in range(0, num_epochs):
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[1]* x.shape[2], batch_size)
            x = x.split(10000, dim=0)[0]
            y = y.transpose(0,1)
            pred = model(x.long(), y)
            # pred = pred[0:].reshape(-1, pred.shape[2])
            # y = y[0:].reshape(-1)
            pred = pred[1:].reshape(-1, pred.shape[2])
            y= y[1:].reshape(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            
            
            loss.backward()
            optimizer.step()
            #print("loss", loss)
        
        print(epoch, loss)

def train_model():
    train_path = 'MLDS_hw2_1_data/training_data'
    TRAIN_LABEL_PATH = os.path.join(train_path, 'training_label.json')
    
    with open(TRAIN_LABEL_PATH) as data_file:    
        y_data = json.load(data_file)

    x_data = {}
    TRAIN_FEATURE_DIR = os.path.join(train_path, 'feat')
    for filename in os.listdir(TRAIN_FEATURE_DIR):
        f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename))
        x_data[filename[:-4]] = f
    

    dataset = MLDSDataset(train_path, x_data, y_data, 5)
    
    hyper_param = {
        'batch_size': 50,
        'n_epochs':15,
        'optimizer': 'Adam',
        'optim_param': {
            'lr': 0.01,
        }
    }
    train_dataloader = DataLoader(dataset=dataset, batch_size=hyper_param['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
    pad_idx = dataset.vocab.string_to_index["<pad>"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    encoder = EncoderRNN((10000), 30, 128, 2, 0.5).to(device)
    decoder = DecoderRNN(dataset.vocab_length(), 30, 128, dataset.vocab_length(), 4, 0.5).to(device)
    model = EncodeToDecode(encoder, decoder, dataset.vocab_length(), device)
    model = model.to(device)
    optimizer = getattr(torch.optim, 'Adam')(
        model.parameters(), 
        **hyper_param['optim_param']
    )
    
    train(train_dataloader, model, loss_fn, optimizer, hyper_param['n_epochs'], hyper_param['batch_size'])
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(model.state_dict(), "models/seq2seq5.pth")

if __name__ == '__main__':
    train_model()
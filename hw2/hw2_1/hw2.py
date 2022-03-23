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

device = torch.device('cuda:15' if torch.cuda.is_available() else 'cpu')

TRAIN=False

# class Caption:
#     def __init__(self, name):
#         self.name = name
#         self.trimmed = False
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {PAD_token: "PAD", BOS_token: "BOS", EOS_token: "EOS", UNK_token: "UNK"}
#         self.num_words = 3  # Count SOS, EOS, PAD

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.num_words
#             self.word2count[word] = 1
#             self.index2word[self.num_words] = word
#             self.num_words += 1
#         else:
#             self.word2count[word] += 1

#     # Remove words below a certain count threshold
#     def trim(self, min_count):
#         if self.trimmed:
#             return
#         self.trimmed = True
#         keep_words = []
#         for k, v in self.word2count.items():
#             if v >= min_count:
#                 keep_words.append(k)

#         print('keep_words {} / {} = {:.4f}'.format(
#             len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
#         ))
#         # Reinitialize dictionaries
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {PAD_token: "PAD", BOS_token: "BOS", EOS_token: "EOS", UNK_token, "UNK"}
#         self.num_words = 3 # Count default tokens
#         for word in keep_words:
#             self.addWord(word)


# # Lowercase and remove non-letter characters
# def normalizeString(s):
#     s = s.lower()
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s


# Takes string sentence, returns sentence of word indexes

def train(dataloader, model, loss_fn, optimizer, num_epochs, batch_size):
    model.train() 
    for epoch in range(0, num_epochs):
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[1]* x.shape[2], batch_size)
            
            x = x.split(10000, dim=0)[0]
            #y = y.transpose(0,1)
           
            pred = model(x.long(), y)
            pred = pred[0:].reshape(-1, pred.shape[2])
          
            y = y[0:].reshape(-1)
        
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            
            print("loss", loss)
        
        print(epoch)

def test(test_dataloader, model, batch_size, dataset, out_file):
    with torch.no_grad():
        count = 0
        bleu_sum = 0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[1]* x.shape[2], batch_size)
            x = x.split(10000, dim=0)[0]
            y = y.transpose(0,1)
            pred = model(x.long(), y)
            pred = pred[1:].reshape(-1, pred.shape[2])
            y = y[1:].reshape(-1)
            # loop through pred
            # if [0] in pred:
            pred_length = 0
            target_length = 0
            precision = 0
            T2 = torch.Tensor([0, 1, 2, 3]).to(device)
            output = []
            for i in range(0, len(y)-1):
                # torch. set_printoptions(profile="full")
                # torch. set_printoptions(profile="default")
                output.append(torch.argmax(pred[i]))
                print(output[i])
                #print(output.eval())
                if output[i] not in T2:
                    pred_length +=1
                if y[i] not in {0, 1, 2, 3}:
                    target_length +=1
                    if output[i]== y[i]:
                        precision+=1
            
            if pred_length > target_length:
                bp = 1
            else:
                if pred_length == 0:
                    bp = 0
                else:    
                    bp = math.exp(1-target_length/pred_length)
            count+= 1
            bleu = bp * precision
            print(bleu)
            bleu_sum += bleu
            string = "Score: " + str(bleu)
            out_file.write(string)
            #accu_number += torch.sum(pred == y)
        bleu_avg = bleu_sum / count
        print('BLEU: %.4f' % bleu_avg)
       
       


    
def hw2(data_path, output):
    path = data_path.split('/')
    path = path[1].split('_')
    label_file = path[0] + '_label.json'
   
    TEST_LABEL_PATH = os.path.join(data_path, label_file)

    with open(TEST_LABEL_PATH) as data_file:    
        test_y_data = json.load(data_file)

    

    x_data = {}
    TRAIN_FEATURE_DIR = os.path.join(data_path, 'feat')
    for filename in os.listdir(TRAIN_FEATURE_DIR):
        f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename))
        x_data[filename[:-4]] = f

    test_x_data = {}
    TEST_FEATURE_DIR = os.path.join(data_path, 'feat')
    for filename in os.listdir(TEST_FEATURE_DIR):
        f = np.load(os.path.join(TEST_FEATURE_DIR, filename))
        test_x_data[filename[:-4]] = f
    
    # just need to get vocab size of trained model
    #dataset = MLDSDataset(train_path, x_data, y_data)
    vocab_length = 1947
    test_dataset = MLDSDataset(data_path, test_x_data, test_y_data)
   
    encoder = EncoderRNN((10000), 30, 128, 2, 0.5).to(device)
    decoder = DecoderRNN(vocab_length, 30, 128, vocab_length, 4, 0.5).to(device)
    model = EncodeToDecode(encoder, decoder, vocab_length, device)
    model = model.to(device)
    
    model.load_state_dict(torch.load("models/seq2seq5.pth"))
    model.eval()
    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=True)
    f = open(sys.argv[2], "a")
    f.write("Bleu scores")
    test(test_dataloader, model, batch_size=1, dataset=test_dataset, out_file=f)
    f.close()


if __name__ == '__main__':
    print(sys.argv[0])
    print(sys.argv[1])
    hw2(sys.argv[1], sys.argv[2])

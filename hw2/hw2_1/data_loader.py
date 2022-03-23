import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import nltk
#from nltk import regexp_tokenize

MAX_LENGTH = 10

class Vocab:
    def __init__(self, freq_threshold=3):
        self.index_to_string = {0:"<pad>", 1:"<bos>", 2:"<eos>", 3:"<unk>"}
        self.string_to_index = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        self.freq_threshold=freq_threshold
        self.words = 4

    def tokenizer(self, sentence):
        #return nltk.word_tokenize(sentence)
        return nltk.regexp_tokenize(sentence, pattern=r"\s|[\.,;]", gaps=True)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                # ignore special tokens
                if word != "<pad>" and word != "<bos>" and word != "<eos>":
                    if word not in frequencies:
                        frequencies[word] = 1
                        
                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.string_to_index[word] = idx
                        self.index_to_string[idx] = word
                        idx += 1
                        self.words += 1

    def numericalize(self, sentence):
        tokens = self.tokenizer(sentence)
        return [
            [self.string_to_index[token]] if token in self.string_to_index else [self.string_to_index["<unk>"]]
            for token in tokens
        ]
    
    def denumericalize(self, array):
        return [
            [self.index_to_string[value]]
            for value in array
        ]

class MLDSDataset(Dataset):
    def __init__(self, root_dir, x_data, y_data, transform=None, freq_threshold=3):
        self.root_dir = root_dir
        self.df = y_data
        self.transform = transform

        self.captions = []
        #self.video_ids = x_data.keys()
        self.video_ids = []
        self.videos = x_data
        for y in y_data:
            for caption in y['caption']:
                # only use captions with less than 10 words so we can easily pad all captions to have a length of 10
                if len(caption.split()) <= 10 and len(caption.split()) > 6: 
                    caption = caption.rstrip(',')
                    caption = caption.rstrip('.')
                    caption = "<bos> " + caption
                    # handles a specific case label case
                    if ',' in caption:
                        caption = caption.replace(',', ' ')
                    while len(caption.split()) <= 10:
                        caption = caption + " <pad>"
                    caption = caption + " <eos>"
                    self.captions.append(caption)
                    self.video_ids.append(y['id'])

        self.vocab = Vocab(freq_threshold)
        self.vocab.build_vocabulary(self.captions)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        
        video_id = self.video_ids[index]
        video = self.videos[video_id]
       
        #print(self.vocab.numericalize("deal"))
        numericalized_caption = self.vocab.numericalize(caption)
       
        return video, torch.squeeze(torch.tensor(numericalized_caption))
    
    def vocab_length(self):
        return self.vocab.words
    
    def get_vocab(self):
        return self.vocab
import pandas as pd
import spacy 
import pickle
from config import config

spacy_vocab_en = spacy.load('en_core_web_sm')

class Vocab_Builder:
    
    def __init__ (self):

        # freq_threshold is to allow only words with a frequency higher 
        # than the threshold

        self.itos = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}  #index to string mapping
        self.stoi = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}  # string to index mapping

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        #Removing spaces, lower, general vocab related work

        return [token.text.lower() for token in spacy_vocab_en.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {} # dict to lookup for words
        idx = 4

        # FIXME better ways to do this are there
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1 
                if(frequencies[word] == 1):
                    #Include it
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # Convert text to numericalized values
    def numericalize(self,text):
        tokenized_text = self.tokenizer(text) # Get the tokenized text
        
        # Stoi contains words which passed the freq threshold. Otherwise, get the <UNK> token
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
        for token in tokenized_text ]
    
    def denumericalize(self, token):
        text = [self.itos[token] if token in self.itos else self.itos[3]]
        return text

    def __repr__(self) -> str:
        arr=''
        for key,value in self.itos.items():
            arr += str(key) + (' ') + str(value) + ('\n')
        return arr


def serialize():
    # data_location =  "flickr8k"
    # caption_file = 'flickr8k/captions.txt'

    vocabulary = Vocab_Builder()

    df = pd.read_csv(config['caption_file_path'])

    captions = df["caption"]

    split_factor = 38003

    captions = captions[0:split_factor]

    vocabulary.build_vocabulary(captions.tolist())

    print(len(vocabulary))
    
    print(vocabulary)

    with open('vocab.pickle', 'wb') as f:
        pickle.dump(vocabulary, f)
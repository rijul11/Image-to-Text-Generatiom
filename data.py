import os
import pandas as pd
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from vocabulary import *
from config import config

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, test, transform = None, freq_threshold = 2):
        self.root_dir = root_dir
        self.df = pd.read_csv(config['caption_file_path'])
        self.transform = transform
        # Get images, caption column from pandas
        split_factor = 38002 # 4000/ 5 = reserving ~200 images for testing
        
        self.imgs = self.df["image"]
        self.imgs_test = self.imgs[split_factor:]
        self.imgs = self.imgs[0:split_factor]
        self.captions = self.df["caption"]
        self.captions_test = self.captions[split_factor:]
        self.captions = self.captions[0:split_factor]
        self.test = test
        self.vocab = Vocab_Builder(freq_threshold) # freq threshold is experimental
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __len__(self):
        if (self.test == True):
            return len(self.imgs_test)
        
        return len(self.imgs)
    
    def __getitem__(self, index: int):

        # Indices are randomly sampled if Shuffle = True
        # otherwise sequentially.

        if self.test == False:
            caption = self.captions[index]
            img_id = self.imgs[index]
        elif self.test == True:
            index += 38002
            caption = self.captions_test[index]
            img_id = self.imgs_test[index]
           
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]] # stoi is string to index, start of sentence
        numericalized_caption += self.vocab.numericalize(caption) # Convert each word to a number in our vocab
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
    
    def __repr__(self) -> str:
        return self.vocab.iloc[1000]
        
    @staticmethod
    def evaluation(self, index : int):
        caption = self.captions_test[index]
        img_id = self.imgs_test[index]
        # print('3')
        # Read the image corresponding to the index
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        # print(img)
        if self.transform is not None:
            img = self.transform(img)
        
        # Fixed BLEU score evaluation
        caption = self.vocab.tokenizer(caption)
        # print(caption)
        return img, caption
    

class Collate:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        
        lengths = torch.tensor([len(cap) for cap in targets]).long()
        
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        # I did not do batch_first = False in the beginning so I had to use torch.permute(1,0)
        # It ensures that shape : (batch_size, max_caption_length) 
        
        return imgs, targets, lengths

# caption file, Maybe change num_workers

def get_loader( root_folder,annotation_file, transform, batch_size = 32,  num_workers = 8, shuffle = True, pin_memory = False, test = False):
    

    dataset =  Flickr8kDataset(root_folder, annotation_file, test, transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn =  Collate(pad_idx = pad_idx)
    )

    return loader, dataset
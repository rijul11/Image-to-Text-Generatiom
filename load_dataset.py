import os
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class Flickr8kDataset(Dataset):
    
    def __init__(self,config,path,training=True):
        
        with open(path,"r") as f:
            self.data = [line.replace("\n", "") for line in f.readlines()]
        
        self.training = training
        self.inference_captions = self.group_captions(self.data)
        
        with open(config["token2idx_path"],"rb") as f:
            self.token2idx = pickle.load(f)
        self.idx2token = {str(idx): token for token, idx in self.token2idx.items()}
        
        self.start_idx = config["START_idx"]
        self.end_idx = config["END_idx"]
        self.pad_idx = config["PAD_idx"]
        self.UNK_idx = config["UNK_idx"]
        # Auxiliary token marks
        self.START_token = config["START_token"]
        self.END_token = config["END_token"]
        self.PAD_token = config["PAD_token"]
        self.UNK_token = config["UNK_token"]
        
        self.max_len = config["max_len"]
        
        self.image_specs = config["image_specs"]
        self.image_transform = self.construct_image_transform(self.image_specs["image_size"])

        # Create paths to image files belonging to the subset
        subset = "train" if training else "validation"
        self.image_dir = self.image_specs["image_dir"][subset]

        # Create (X, Y) pairs
        self.data = self.create_input_label_mappings(self.data)

        self.dataset_size = len(self.data) if self.training else 0
    
    def __len__(self):
        return self.dataset_size
    
    def __refr__(self):
        return self.data
    
    def construct_image_transform(self, image_size):
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        
        return preprocessing
    
    def load_and_process_images(self, image_dir, image_names):
       
        image_paths = [os.path.join(image_dir, fname) for fname in image_names]
        # Load images
        images_raw = [Image.open(path) for path in image_paths]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        image_tensors = [self.image_transform(img) for img in images_raw]

        images_processed = {img_name: img_tensor for img_name, img_tensor in zip(image_names, image_tensors)}
        return images_processed
    
    def group_captions(self, data):
       
        grouped_captions = {}

        for line in data:
            caption_data = line.split()
            img_name, img_caption = caption_data[0].split("#")[0], caption_data[1:]
            if img_name not in grouped_captions:
                # We came across the first caption for this particular image
                grouped_captions[img_name] = []

            grouped_captions[img_name].append(img_caption)

        return grouped_captions
    
    def create_input_label_mappings(self, data):
       
        processed_data = []
        for line in data:
            tokens = line.split()
            # Separate image name from the label tokens
            img_name, caption_words = tokens[0].split("#")[0], tokens[1:]
            # Construct (X, Y) pair
            pair = (img_name, caption_words)
            processed_data.append(pair)

        return processed_data
    
    def load_and_prepare_image(self, image_name):
        
        image_path = os.path.join(self.image_dir, image_name)
        img_pil = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(img_pil)
        return image_tensor
    
    def inference_batch(self, batch_size):
        
        caption_data_items = list(self.inference_captions.items())
        # random.shuffle(caption_data_items)

        num_batches = len(caption_data_items) // batch_size
        for idx in range(num_batches):
            caption_samples = caption_data_items[idx * batch_size: (idx + 1) * batch_size]
            batch_imgs = []
            batch_captions = []

            # Increase index for the next batch
            idx += batch_size

            # Create a mini batch data
            for image_name, captions in caption_samples:
                batch_captions.append(captions)
                batch_imgs.append(self.load_and_prepare_image(image_name))

            # Batch image tensors
            batch_imgs = torch.stack(batch_imgs, dim=0)
            if batch_size == 1:
                batch_imgs = batch_imgs.unsqueeze(0)

            yield batch_imgs, batch_captions
            
    def __getitem__(self, index):
        # Extract the caption data
        img_id, tokens = self.data[index]

        # Load and preprocess image
        image_tensor = self.load_and_prepare_image(img_id)

        # Pad the token and label sequences
        tokens = tokens[:self.max_len]

        tokens = [token.strip().lower() for token in tokens]
        tokens = [self.START_token] + tokens + [self.END_token]
        # Extract input and target output
        input_tokens = tokens[:-1].copy()
        tgt_tokens = tokens[1:].copy()

        # Number of words in the input token
        sample_size = len(input_tokens)
        padding_size = self.max_len - sample_size

        if padding_size > 0:
            padding_vec = [self.PAD_token for _ in range(padding_size)]
            input_tokens += padding_vec.copy()
            tgt_tokens += padding_vec.copy()

        # Apply the vocabulary mapping to the input tokens
        input_tokens = [self.token2idx.get(token, self.UNK_idx) for token in input_tokens]
        tgt_tokens = [self.token2idx.get(token, self.UNK_idx) for token in tgt_tokens]

        input_tokens = torch.Tensor(input_tokens).long()
        tgt_tokens = torch.Tensor(tgt_tokens).long()

        # Index from which to extract the model prediction
        # Define the padding masks
        tgt_padding_mask = torch.ones([self.max_len, ])
        tgt_padding_mask[:sample_size] = 0.0
        tgt_padding_mask = tgt_padding_mask.bool()

        return image_tensor, input_tokens, tgt_tokens, tgt_padding_mask
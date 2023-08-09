import os
import string
import pickle
import numpy as np
import gensim.downloader as api

def preprocess_caption(caption):
    # Store punctuation in a table
    table = str.maketrans("", "", string.punctuation)
    # Split the caption into tokens
    caption = caption.split()
    # Convert tokens to lower case
    caption = [token.lower() for token in caption]
    # Remove punctuations
    caption = [token.translate(table) for token in caption]
    # Remove tokens of length 1 like "'s" or "a"
    caption = [token for token in caption if len(token)>1]
    # Remove numerical values
    caption = [token for token in caption if token.isalpha()]
    return " ".join(caption)


def save_captions(img_caption, img_batch, save_path):
    with open(save_path, 'w') as f:
        for img_name in img_batch:
            img_id = os.path.splitext(img_name)[0]
            #print(img_id)
            if img_id in img_caption:
                for caption in img_caption[img_id]:
                    f.write("{} {}\n".format(img_name, caption))
        

def split_dataset(img_caption, split_paths, save_paths):
    for load_path, save_path in zip(split_paths, save_paths):
        with open(load_path, 'r') as f:
            img_batch = [img.replace("\n", "") for img in f.readlines()]
        save_captions(img_caption, img_batch, save_path)
        

def extract_embeddings(config, vocab):
    np.random.seed(config["seed"])
    embeddings_config = config["embeddings"]
    word2vec_model = api.load('word2vec-google-news-300')
    save_path_emb = embeddings_config["path"]
    embedding_dim = embeddings_config["size"]

    punct_table = str.maketrans("", "", string.punctuation)

    # Used for finding the embedding vector for each token
    vectors = []
    new_vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    # Counter used for determining the mapping from word to index
    i = len(new_vocab)

    # embedding_file_name = "glove.6B.{}d.txt".format(embedding_dim)
    # embeddings_path = os.path.join(config["glove_dir"], embedding_file_name)
    # with open(embeddings_path, "rb") as f:
    for token in word2vec_model.key_to_index:
        # token = token.strip().lower()
        # token = token.translate(punct_table)
        if token in vocab and token not in new_vocab:
            # Save embedding only for words present in the vocab
            #print(token)
            embedding_vec = [word2vec_model[token]]
            #print(embedding_vec)
            vectors += embedding_vec
            new_vocab[token] = i
            i += 1
            
    with open(config["token2idx_path"], 'wb') as f:
        pickle.dump(new_vocab, f)
        
    vectors = np.array(vectors)
    # Embedding vector for tokens used for padding the input sequence
    pad_embedding = np.zeros((embedding_dim,))
    # Embedding vector for start of the sequence
    sos_embedding = np.random.normal(size=(embedding_dim,))
    # Embedding vector for end of the sequence
    eos_embedding = np.random.normal(size=(embedding_dim,))
    # Embedding vector for unknown token
    unk_embedding =  np.random.normal(size=(embedding_dim,))
    
    vectors = np.vstack((pad_embedding, sos_embedding, eos_embedding, unk_embedding, vectors))
    # Save extracted embeddings
    np.savetxt(save_path_emb, vectors)
            

def create_vocab(img_caption):
    # Vocabulary dictionary
    token2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    # All possible words in the token
    tokens = set()
    # Extract all tokens from the image captions
    for captions in img_caption.values():
        current_tokens = [token for caption in captions for token in caption.split()]
        tokens.update(current_tokens)

    starting_len = len(token2idx)
    tokens = list(tokens)
    token2idx.update({token: (idx + starting_len) for idx, token in enumerate(tokens)})
    
    print(len(token2idx))

    return token2idx


def clean_captions(id2annotation):
    img_caption_clean = id2annotation.copy()
    for img_id, captions in id2annotation.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Preprocess caption
            clean_caption = preprocess_caption(caption)
            # Save the cleaned caption
            img_caption_clean[img_id][i] =  clean_caption

    return img_caption_clean


def load_captions(data):
    image2caption = dict()
    for sample in data.split("\n"):
        tokens = sample.split()
        if len(sample) < 2:
            # Image has no description: Invalid data row
            continue
		# First token is image id, remaining ones correspond to the caption
        img_name, img_caption = tokens[0], tokens[1:]

        img_id = img_name.split(".")[0]
        # Recreate the description
        img_caption = " ".join(img_caption)
        
        if img_id not in image2caption:
            image2caption[img_id] = list()
        # Save the description
        image2caption[img_id].append(img_caption)

    return image2caption
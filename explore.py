import gensim.downloader as api
import numpy as np

word2vec_model = api.load('word2vec-google-news-300')

print(len(word2vec_model))

token_embeddings = {}

for token in word2vec_model.key_to_index:
    token_embeddings[token]= word2vec_model[token]
print(token_embeddings)
    
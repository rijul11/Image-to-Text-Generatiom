import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torchvision.models import resnet50, ResNet50_Weights

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=16):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(weights = ResNet50_Weights.DEFAULT, pretrained=True) 

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


class ResidualBlock(nn.Module):
    
    def __init__(self, input_dim):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim,input_dim)
        )
        
    def forward(self, x):
        
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x
    
    
class Normalize(nn.Module):
    
    def __init__(self, eps = 1e-5):
        super(Normalize, self).__init__()
        self.register_buffer("eps", torch.Tensor([eps]))
        
    def forward(self, x, dim = -1):
        norm = x.norm(2, dim = dim).unsqueeze(-1)
        x = self.eps * (x / norm)
        return x
    
    
class PositionalEncodings(nn.Module):

    def __init__(self, seq_len, d_model, p_dropout):
        
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x
        
        
class CaptionDecoder(nn.Module):
    
    def __init__(self, config):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        model_config = config["model_configuration"]
        decoder_layers = model_config["decoder_layers"]
        attention_heads = model_config["attention_heads"]
        d_model = model_config["d_model"]
        ff_dim = model_config["ff_dim"]
        dropout = model_config["dropout"]

        embedding_dim = config["embeddings"]["size"]
        vocab_size = config["vocab_size"]
        img_feature_channels = config["image_specs"]["img_feature_channels"]

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        self.entry_mapping_words = nn.Linear(embedding_dim, d_model)
        self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.res_block = ResidualBlock(d_model)

        self.positional_encodings = PositionalEncodings(config["max_len"], d_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            activation="relu",
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, tgt_padding_mask=None, tgt_mask=None):
        """Performs forward pass of the module."""
        # Adapt the dimensionality of the features for image patches
        image_features = self.entry_mapping_img(image_features)
        image_features = image_features.permute(1, 0, 2)
        image_features = F.leaky_relu(image_features)

        # Entry mapping for word tokens
        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = F.leaky_relu(x)

        x = self.res_block(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(
            tgt=x,
            memory=image_features,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x
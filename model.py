import math
from typing import List, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, num_embeddings, embed_dim, max_len=512, pad_idx=0):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=pad_idx
        )
        pe = torch.zeros((max_len, hidden_size))
        position = torch.arange(0, max_len, dtype=torch.float32)[:, None]
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0)) / hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pe[:, :x.shape[1], :]
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, input_dim, output_dim, norm=None, ):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for i in range(self.num_layers):
            out = self.layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out


class SCNet(nn.Module):
    def __init__(self, block_layers, num_channel, input_dim, output_dim, embed_dim, kernel_1, kernel_2, kernel_3):
        super(SCNet, self).__init__()
        self.block_layers = block_layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=(kernel_1, embed_dim))
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_2, stride=3, padding=2)
        self.conv2 = nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernel_3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernel_3, stride=1, padding=1)
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.conv1(x))
        out = out.squeeze(3)
        for i in range(self.block_layers):
            out = F.relu(self.pool1(out))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
        out = F.max_pool1d(out, out.shape[2])
        out = self.fc(out.reshape(-1, self.input_dim))
        return out


class DL4SC(nn.Module):
    def __init__(self, d_model=512, n_head=16, batch_first=True, num_layers=10, input_dim=1000, out_dim=10, v_size=1000,
                 block_layers=12, kernel_1=3, kernel_2=4, kernel_3=3):
        super(DL4SC, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_size=d_model, num_embeddings=v_size, embed_dim=d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers,
                                                      input_dim=input_dim, output_dim=out_dim)
        self.Scnet = SCNet(num_channel=d_model, embed_dim=d_model, kernel_1=kernel_1, kernel_2=kernel_2,
                           kernel_3=kernel_3, block_layers=block_layers,input_dim=input_dim, output_dim=out_dim)

    def forward(self, x, mask=None):
        out = self.positional_encoding(x)
        output = self.transformer_encoder(out, mask=mask)
        output = self.Scnet(output)
        return output

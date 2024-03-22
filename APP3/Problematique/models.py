# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        self.embeding = nn.Embedding(self.dict_size['seq'], self.hidden_dim)
        self.encoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size['target'])
        self.to(device)

    def forward(self, x):
        # Encoder
        out_enc, hidden = self.encoder_layer(self.embeding(x))

        # Decoder
        max_len = self.maxlen['target']
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['target'])).to(self.device)

        # Boucle pour les les symboles de sortie
        for i in range(max_len):
            out_dec, hidden = self.decoder_layer(vec_in, hidden)
            output = self.fc(out_dec)
            vec_out[:][i][:] = output.squeeze(1)

            vec_in = output.argmax(dim=2)

        return vec_out, hidden
    


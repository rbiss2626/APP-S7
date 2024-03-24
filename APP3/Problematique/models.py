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
        self.embeding_decoder = nn.Embedding(self.dict_size, self.hidden_dim)
        
        self.encoder_layer = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Couches pour attention
        self.att_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.att_softmax = nn.Softmax()
        self.att_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)
            
        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)  
        self.to(device)

    def forward(self, x):
        # Encoder
        out_enc, hidden = self.encoder_layer(x)

        # Decoder
        max_len = self.maxlen['target']
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)
        

        # Boucle pour les les symboles de sortie
        for i in range(max_len):
            out_dec, hidden = self.decoder_layer(self.embeding_decoder(vec_in), hidden)
            
            # Calcul de l'attention
            query = self.att_hidden(out_dec)                
            similarity = torch.matmul(out_enc, query.transpose(1, 2))
            att_w = self.att_softmax(similarity).transpose(1, 2)
            attention = torch.bmm(att_w, out_enc)
            out_att = torch.cat((out_dec, attention), dim=2)
            out_att = self.att_combine(out_att)
            
            output = self.fc(out_att)
            vec_out[:, i, :] = output.squeeze(1)

            vec_in = output.argmax(dim=2)

        return vec_out, hidden
    


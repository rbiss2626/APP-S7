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
        self.max_len = maxlen

        # Definition des couches
        # Couches pour rnn
        self.target_embedding = nn.Embedding(self.dict_size, self.hidden_dim)
        # self.seq_embedding = nn.Embedding(self.dict_size['seq'], self.hidden_dim)
        self.encoder_layer = nn.GRU(2, self.hidden_dim, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size) #on sort un one-hot pour les lettres

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn
    

    def encoder(self, x):
        # x = torch.unsqueeze(x, -1).float()

        out, hidden = self.encoder_layer(x) # pas d'embeding, on a déja des nombres
        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['target'] 
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device) # Vecteur de sortie du décodage et prochaine entrée

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            out, hidden = self.decoder_layer(self.target_embedding(vec_in), hidden) # ici on fait l'embedding, car on a la lettre précédente en entrée
            output = self.fc(out)
            vec_out[:, i, :] = output.squeeze(1)
            vec_in = output.argmax(dim=2) # on sort l'index de la valeur la plus haute

        return vec_out, hidden, None


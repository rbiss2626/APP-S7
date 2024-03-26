# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, with_attention=False):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.with_attention = with_attention

        # Definition des couches
        # Couches pour rnn
        self.embeding_decoder = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_layer = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Couches pour attention
        self.att_softmax = nn.Softmax()
        self.att_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.att_hidden = nn.Linear(self.hidden_dim, self.hidden_dim) 
            
        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)  
        self.to(device)
    
    def attention (self, out_dec, out_enc):
        query = self.att_hidden(out_dec)                
        similarity = torch.matmul(out_enc, query.transpose(1, 2))
        att_w = self.att_softmax(similarity).transpose(1, 2)
        attention = torch.bmm(att_w, out_enc)
        out_att = torch.cat((out_dec, attention), dim=2)
        out_att = self.att_combine(out_att)
        return out_att, att_w
        
    def encoder (self, x):
        # Encoder
        out_enc, hidden = self.encoder_layer(x)
        return out_enc, hidden
    
    def decoder (self, vec_in, hidden, out_enc):
        #Decoder
        out_dec, hidden = self.decoder_layer(self.embeding_decoder(vec_in), hidden)
        return out_dec, hidden
    
    def decoder_att (self, vec_in, hidden, out_enc):
        #Decoder
        out_dec, hidden = self.decoder_layer(self.embeding_decoder(vec_in), hidden)
        out_att, att_w = self.attention(out_dec, out_enc)
        return out_att, hidden, att_w
        
    def forward(self, x):
        # Encoder
        out_enc, hidden = self.encoder(x)

        # Decoder
        max_len = self.maxlen['target']
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)
        att_w_out = torch.zeros((batch_size, max_len, x.shape[1])).to(self.device)
        
        # Boucle pour les les symboles de sortie
        for i in range(max_len):
            if self.with_attention:
                out_att, hidden, att_w = self.decoder_att(vec_in, hidden, out_enc)
            else:
                out_dec, hidden = self.decoder(vec_in, hidden, out_enc)
                out_att = out_dec
                att_w = torch.zeros((batch_size, x.shape[1], 1)).to(self.device)
            
            output = self.fc(out_att)
            vec_out[:, i, :] = output.squeeze(1)
            att_w_out[:, i, :] = att_w.squeeze(1)

            vec_in = output.argmax(dim=2)

        return vec_out, hidden, att_w_out
    
class trajectory2seq_bi(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, with_attention=False):
        super(trajectory2seq_bi, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.with_attention = with_attention

        # Definition des couches
        # Couches pour rnn
        self.embeding_decoder = nn.Embedding(self.dict_size, self.hidden_dim)
        self.encoder_layer = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc_hidden = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        # Couches pour attention
        self.att_softmax = nn.Softmax()
        
        self.att_combine = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.att_hidden = nn.Linear(self.hidden_dim, self.hidden_dim*2)       

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)  
        self.to(device)
    
    def attention (self, out_dec, out_enc):
        query = self.att_hidden(out_dec)                
        similarity = torch.matmul(out_enc, query.transpose(1, 2))
        att_w = self.att_softmax(similarity).transpose(1, 2)
        attention = torch.bmm(att_w, out_enc)
        out_att = torch.cat((out_dec, attention), dim=2)
        out_att = self.att_combine(out_att)
        return out_att, att_w
            
    def encoder_bi (self, x):
        # Encoder
        out_enc, hidden = self.encoder_layer(x)
        
        hidden = hidden.reshape(-1, self.hidden_dim*2)
        hidden = self.fc_hidden(hidden)
        hidden = hidden.reshape(self.n_layers, x.shape[0], self.hidden_dim)   
        
        return out_enc, hidden
    
    def decoder (self, vec_in, hidden, out_enc):
        #Decoder
        out_dec, hidden = self.decoder_layer(self.embeding_decoder(vec_in), hidden)
        return out_dec, hidden
    
    def decoder_att (self, vec_in, hidden, out_enc):
        #Decoder
        out_dec, hidden = self.decoder_layer(self.embeding_decoder(vec_in), hidden)
        out_att, att_w = self.attention(out_dec, out_enc)
        return out_att, hidden, att_w
        
    def forward(self, x):
        # Encoder
        out_enc, hidden = self.encoder_bi(x)

        # Decoder
        max_len = self.maxlen['target']
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)
        att_w_out = torch.zeros((batch_size, max_len, x.shape[1])).to(self.device)
        
        # Boucle pour les les symboles de sortie
        for i in range(max_len):
            if self.with_attention:
                out_att, hidden, att_w = self.decoder_att(vec_in, hidden, out_enc)
            else:
                out_dec, hidden = self.decoder(vec_in, hidden, out_enc)
                out_att = out_dec
                att_w = torch.zeros((batch_size, x.shape[1], 1)).to(self.device)
            
            output = self.fc(out_att)
            vec_out[:, i, :] = output.squeeze(1)
            att_w_out[:, i, :] = att_w.squeeze(1)

            vec_in = output.argmax(dim=2)

        return vec_out, hidden, att_w_out
    
    
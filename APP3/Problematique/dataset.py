import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle
import os

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.join(dir_path, filename)
        with open(dir_path, 'rb') as fp:
            self.data = pickle.load(fp)

        # Extraction des symboles
        self.symb2int = {start_symbol:0, stop_symbol:1, pad_symbol:2}
        cpt_symb_target = 3

        for i in range(len(self.data)):
            target, seq = self.data[i]

            target = list(target)
            for symb in target:
                if symb not in self.symb2int:
                    self.symb2int[symb] = cpt_symb_target
                    cpt_symb_target += 1
            self.data[i][0] = target

        # Ajout du padding aux séquences
        self.max_len = dict()

        self.max_len['target'] = 0
        for i in range(len(self.data)):
            if len(self.data[i][0]) > self.max_len['target']:
                self.max_len['target'] = len(self.data[i][0])
        self.max_len['target'] += 1

        self.max_len['seq'] = 0
        for i in range(len(self.data)):
            if len(self.data[i][1][0]) > self.max_len['seq']:
                self.max_len['seq'] = len(self.data[i][1][0])
        self.max_len['seq'] += 1

        for i in range(len(self.data)):
            self.data[i][0] += [self.stop_symbol] + [self.pad_symbol] * (self.max_len['target'] - len(self.data[i][0]) - 1)
            
        self.int2symb = {v for v in self.symb2int}

        for i in range(len(self.data)):
            data_x = self.data[i][1][0][-1]
            data_y = self.data[i][1][1][-1]

            pad_seq_x = np.array([data_x] * (self.max_len['seq'] - len(self.data[i][1][0]) - 1))
            pad_seq_y = np.array([data_y] * (self.max_len['seq'] - len(self.data[i][1][1]) - 1))
            
            new_arr_x = np.concatenate((self.data[i][1][0], pad_seq_x))
            new_arr_y = np.concatenate((self.data[i][1][1], pad_seq_y))

            self.data[i][1] = np.array([new_arr_x, new_arr_y])
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def visualisation(self, idx):
        # Visualisation des échantillons
        title = self.data[idx][0]
        x, y = self.data[idx][1]

        plt.scatter(x, y)
        plt.title(title)
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
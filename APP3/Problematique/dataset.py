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

        for i in range(len(self.data)):
            self.data[i][0] += [self.stop_symbol] + [self.pad_symbol] * (self.max_len['target'] - len(self.data[i][0]) - 1)
            
        self.int2symb = {v:k for k,v in self.symb2int.items()}

        for i in range(len(self.data)):
            data_x = 0
            data_y = 0
            
            self.data[i][1] = torch.diff(torch.tensor(self.data[i][1]), dim=1).cpu().detach().numpy()
            
            pad_sequence_x = np.array([data_x] * (self.max_len['seq'] - self.data[i][1].shape[1] - 1))
            pad_sequence_y = np.array([data_y] * (self.max_len['seq'] - self.data[i][1].shape[1] - 1))
            
            temp_x = np.concatenate((self.data[i][1][0], pad_sequence_x))
            temp_y = np.concatenate((self.data[i][1][1], pad_sequence_y))
            
            self.data[i][1] = np.array([temp_x, temp_y])
        
        self.dict_size = len(self.int2symb)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataSeq = self.data[idx][1]
        target = self.data[idx][0]
        
        target = [self.symb2int[i] for i in target]

        return torch.tensor(np.array(dataSeq)), torch.tensor(target)

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

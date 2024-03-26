import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)
        
        
        # Extraction des symboles
        self.symb2int = {start_symbol:0, stop_symbol:1, pad_symbol:2}
        cpt_symb = 3
        for i in range(len(self.data)):
            self.data[i][0] = list(self.data[i][0])
            for symb in self.data[i][0]:
                if symb not in self.symb2int:
                    self.symb2int[symb] = cpt_symb
                    cpt_symb += 1
            

        # dictionnaires d'entiers vers symboles 
        self.int2symb = dict()
        self.int2symb = {v:k for k,v in self.symb2int.items()} 

        self.maxLen = dict()        

        # Ajout du padding aux séquences

        # -- target --
        # on fait comme au lab, on trouve la longueur max et on pad avec des <pad>
        self.maxLen['target'] = 0
        for i in range(len(self.data)):
            if len(self.data[i][0]) > self.maxLen['target']:
                self.maxLen['target'] = len(self.data[i][0])
        self.maxLen['target'] += 1 # pour le EOS

        for i in range(len(self.data)):
            self.data[i][0] += [self.stop_symbol] + [self.pad_symbol]*(self.maxLen['target'] - len(self.data[i][0]) -1)

        # ----------------

        #  -- sequence --
        # ici, on trouve la longueur max et on pad avec le dernier chiffre qu'on répete.
        self.maxLen['seq'] = 0
        for i in range(len(self.data)):
            if len(self.data[i][1][0]) > self.maxLen['seq']:
                self.maxLen['seq'] = len(self.data[i][1][0])
        
        for i in range(len(self.data)):
            self.data[i][1] = torch.diff(torch.tensor(self.data[i][1]), dim=1).cpu().detach().numpy()

            data_x = 0
            data_y = 0

            pad_seq_x = np.array([data_x] * ((self.maxLen['seq'] - len(self.data[i][1][0])) - 1))
            pad_seq_y = np.array([data_y] * ((self.maxLen['seq'] - len(self.data[i][1][1])) - 1))

            tempX = np.concatenate((self.data[i][1][0], pad_seq_x))  
            tempY = np.concatenate((self.data[i][1][1], pad_seq_y))

            self.data[i][1] = np.array([tempX, tempY])

        # ----------------

        self.dictSize = len(self.int2symb) # nombre total de symbole

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataSeq = self.data[idx][1]
        targetSeq = self.data[idx][0]

        targetSeq = [self.symb2int[i] for i in targetSeq]

        return (torch.tensor(np.array(dataSeq)), torch.tensor(targetSeq))

    def visualisation(self, idx):
        title = self.data[idx][0]
        x, y = self.data[idx][1]

        plt.scatter(x, y)
        plt.title(title)
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        print(a[3])
        a.visualisation(np.random.randint(0, len(a)))
# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 10
    lr = 0.01
    batch_size = 50
    n_hidden = 20
    n_layers = 2

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    _dir_path = os.path.dirname(__file__)
    _dir_path = os.path.join(_dir_path, 'data_trainval.p')
    dataset = HandwrittenWords(_dir_path)

    
    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset,
                                                                [int(len(dataset) * 0.8),
                                                                 int(len(dataset) - int(len(dataset) * 0.8))])
   
    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers) 
    
    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    model = trajectory2seq(n_hidden, n_layers, dataset.int2symb, dataset.symb2int, dataset.max_len, device, dataset.max_len)

    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))
    
    # Initialisation des variables


    if trainning:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            
            for batch_idx, data in enumerate(dataload_train):
                #formatage des donnees
                target, seq = data
                seq = seq.to(device).long()
                # target = target.to(device).long()
                
                optimizer.zero_grad()
                output, hidden = model(seq)
                loss = criterion(output.view((-1, model.dict_size['target'])), target.view(-1))
                
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()
                
                output_list = torch.argmax(output, dim=1).detach().cpu().tolist()
                target_list = target.cpu().tolist()
                M = len(output_list)
                
                for i in range(batch_size):
                    a = target_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1)
                    dist += edit_distance(a[:Ma], b[:Mb])/batch_size
                    
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass
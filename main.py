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
    batch_size = 100            # Taille des lots
    n_epochs = 50               # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur

    n_hidden = 20               # Nombre de neurones caches par couche 
    n_layers = 2               # Nombre de de couches

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')

    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),
                                                                 int(len(dataset)) - int(len(dataset)*0.8)])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0)
    dataload_val = DataLoader(dataset_val,batch_size=batch_size,shuffle=True,num_workers=0)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    model = trajectory2seq(n_hidden,n_layers,dataset.int2symb,dataset.symb2int,dataset.dictSize,device,dataset.maxLen)



    # Initialisation des variables

    # ??

    
    if trainning:

        if learning_curves:
            train_dist =[] # Historique des distances
            train_loss=[] # Historique des coûts
            fig, ax = plt.subplots(1) # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2) # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            running_loss_train = 0
            dist=0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                seq, target = data
                seq = seq.to(device).float()
                seq = torch.swapaxes(seq,1,2)
                target = target.to(device).long()

                optimizer.zero_grad() # Mise a zero du gradient
                output, hidden, attn = model(seq)# Passage avant
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                
                loss.backward() # calcul du gradient
                optimizer.step() # Mise a jour des poids
                running_loss_train += loss.item()
            
                # calcul de la distance d'édition
                output_list = torch.argmax(output,dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1) # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)# longueur mot b
                    dist += edit_distance(a[:Ma],b[:Mb])/batch_size


                # Affichage pendant l'entraînement
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r\n')

            # Validation
            running_loss_val = 0
            model.eval()
            for data in dataload_val:
                in_seq, target_seq = [obj.to(device).float() for obj in data]

                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
                seq, target = data
                seq = seq.to(device).float()
                seq = torch.swapaxes(seq,1,2)
                target = target.to(device).long()

                # optimizer.zero_grad() # Mise a zero du gradient
                output, hidden, attn = model(seq)# Passage avant
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                
                # loss.backward() # calcul du gradient
                # optimizer.step() # Mise a jour des poids
                running_loss_val += loss.item()
            
                # calcul de la distance d'édition
                output_list = torch.argmax(output,dim=-1).detach().cpu().tolist()
                target_seq_list = target.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1) # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)# longueur mot b
                    dist += edit_distance(a[:Ma],b[:Mb])/batch_size


                # Affichage pendant l'entraînement
                print('Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_val.dataset),
                    100. * batch_idx *  batch_size / len(dataload_val.dataset), running_loss_val / (batch_idx + 1),
                    dist/len(dataload_val)), end='\r')
            print('\n')
            # Affichage graphique
            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist/len(dataload_train))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            torch.save(model,'model.pt')

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')

        if gen_test_images:
            model = torch.load('model.pt')
            model.eval()
            dataset.symb2int = model.symb2int
            dataset.int2symb = model.int2symb

            for num in range(10):
                # extraction d'une séquence du dataset de validation
                randNum = np.random.randint(0,len(dataset))
                seq, target = dataset[randNum]
                seq = torch.swapaxes(seq,0,1)
                seq = torch.unsqueeze(seq,0)

                prediction, _, _ = model(seq.float())            
                out = torch.argmax(prediction, dim=2).detach().cpu()[0,:].tolist()
                out_seq = [model.int2symb[i] for i in out]
                print(out_seq[:out_seq.index('<eos>')+1])
                dataset.visualisation(randNum)




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
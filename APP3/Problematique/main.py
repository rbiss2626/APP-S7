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
import os


if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    attention = True            # Attention?
    bidirectionnal = True      # Bidirectionnel?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 50
    lr = 0.008
    batch_size = 50
    n_hidden = 14
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

    _dir_path = os.path.dirname(__file__)
    _dir_path = os.path.join(_dir_path, 'data_test.p')
    dataset_test = HandwrittenWords(_dir_path)

    
    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset,
                                                                [int(len(dataset) * 0.80),
                                                                 int(len(dataset) * 0.2)])
   
    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers) 
    dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('Test data : ', len(dataset_test))
    print('\n')

    # Instanciation du model
    if (bidirectionnal):
        model = trajectory2seq_bi(hidden_dim=n_hidden, n_layers=n_layers, int2symb=dataset.int2symb, symb2int=dataset.symb2int,
                           dict_size=dataset.dict_size, device=device, maxlen=dataset.max_len,
                           with_attention=attention)
    else:
        model = trajectory2seq(hidden_dim=n_hidden, n_layers=n_layers, int2symb=dataset.int2symb, symb2int=dataset.symb2int,
                           dict_size=dataset.dict_size, device=device, maxlen=dataset.max_len,
                           with_attention=attention)
    

    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))
    
    # Initialisation des variables
    train_dist = []
    train_loss = []
    fig, ax = plt.subplots(1)
    
    best_val_dist = -1

    if trainning:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()
            running_loss_train = 0
            dist = 0
            
            for batch_idx, data in enumerate(dataload_train):
                #formatage des donnees
                seq, target = data
                seq = seq.to(device).float()
                seq = torch.swapaxes(seq, 1, 2)
                target = target.to(device).long()
                
                optimizer.zero_grad()
                output, hidden, att = model(seq)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))

                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()
                
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_list = target.cpu().tolist()
                M = len(output_list)
                
                for i in range(batch_size):
                    a = target_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    dist += edit_distance(a[:Ma], b[:Mb])/batch_size
                
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')
                    
            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')
            
            # Validation
            model.eval()
            running_loss_val = 0
            dist_val = 0
            
            for data in dataload_val:
                seq, target = data
                seq = seq.to(device).float()
                seq = torch.swapaxes(seq, 1, 2)
                target = target.to(device).long()
                
                output, hidden, att = model(seq)
                loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
                running_loss_val += loss.item()
                                
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_list = target.cpu().tolist()
                M = len(output_list)
                
                for i in range(batch_size):
                    a = target_list[i]
                    b = output_list[i]
                    Ma = a.index(1)
                    Mb = b.index(1) if 1 in b else len(b)
                    dist_val += edit_distance(a[:Ma], b[:Mb])/batch_size
                
            print('\nValidation - Average loss: {:.4f} Average Edit Distance: {:.4f}'.format(running_loss_val/len(dataload_val), dist_val/len(dataload_val)))
            print('')       
            
            if dist_val/len(dataload_val) < best_val_dist or best_val_dist == -1:
                best_val_dist = dist_val/len(dataload_val)
                torch.save(model, 'model.pt')
                print('Saving new best model\n')

            # Ajouter les loss aux listes
            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(running_loss_val/len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='validation loss')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

        # Affichage
        if learning_curves:
            # visualization
            plt.show()
            plt.close('all')

                

    if test:
        # Évaluation
        model = torch.load('model.pt')
        model.eval()
        dataset_test.symb2int = model.symb2int
        dataset_test.int2symb = model.int2symb
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        
        seq_list_ = []   
        target_list_ = []
        output_list_ = []
        att_list_ = []
        
        target_list_int = []
        output_list_int = []
        
        dist_test = 0
        running_loss_test = 0
        
        for data in dataload_test:
            seq, target = data
            seq = torch.swapaxes(seq, 1, 2)
            seq.unsqueeze(0)
            seq = seq.to(device).float()
            target = target.to(device).long()
            output, hidden, att = model(seq)
            loss = criterion(output.view((-1, model.dict_size)), target.view(-1))
            running_loss_test += loss.item()
            
            output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
            target_list = target.cpu().tolist()
            M = len(output_list)
            
            for i in range(batch_size):
                a = target_list[i]
                b = output_list[i]
                Ma = a.index(1)
                Mb = b.index(1) if 1 in b else len(b)
                dist_test += edit_distance(a[:Ma], b[:Mb])/batch_size
            
            seq_list_.append(seq)
            target_list_.append(target)
            output_list_.append(output)
            att_list_.append(att)

           
        # Affichage des résultats de test
        print('\nTest - Average loss: {:.4f} Average Edit Distance: {:.4f}'.format(running_loss_test/len(dataload_test), dist_test/len(dataload_test)))
        print('')   
                    
        if gen_test_images:
            
            for sublist in target_list_:
                for item in sublist:
                    for letter in item.detach().cpu().tolist():
                        target_list_int.append(letter)
                        
            for sublist in output_list_:
                for item in sublist:
                    for letter in torch.argmax(item, dim=1).detach().cpu().tolist():
                        output_list_int.append(letter)
                      
            matrix = confusion_matrix(target_list_int, output_list_int)
            plt.imshow(matrix,  cmap='binary')
            plt.xticks(np.arange(len(dataset.symb2int)-3), list(dataset.symb2int.keys())[3:])
            plt.yticks(np.arange(len(dataset.symb2int)-3), list(dataset.symb2int.keys())[3:])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            for i in range(10):
                random_batch = np.random.randint(0, len(output_list_))
                random_idx = np.random.randint(0, len(output_list_[random_batch]))
                
                seq = seq_list_[random_batch][random_idx]
                output = output_list_[random_batch][random_idx]
                target = target_list_[random_batch][random_idx]
                att = att_list_[random_batch][random_idx]
                
                output = torch.argmax(output, dim=1).detach().cpu().tolist()
                output = [model.int2symb[i] for i in output]
                
                target = target.detach().cpu().tolist()
                target = [model.int2symb[i] for i in target]
                
                seq = seq.detach().cpu().tolist()
                
                att = att.detach().cpu().numpy()
                        
                x_coord = np.zeros(1)
                y_coord = np.zeros(1)
                
                for i in range(len(seq) - 1):             
                    x_coord = np.append(x_coord, x_coord[-1] + seq[i][0])
                    y_coord = np.append(y_coord, y_coord[-1] + seq[i][1])
                    
                for idx in range(len(att)):
                    x_coord_attn = x_coord[np.argpartition(att[idx], -25)[-25:]]
                    y_coord_attn = y_coord[np.argpartition(att[idx], -25)[-25:]]
          
                    plt.subplot(3,2, idx+1)
                    # plt.scatter(x_coord, y_coord, c=np.array(att[idx]), cmap='gray_r',norm='log')
                    plt.plot(x_coord, y_coord, 'o', color='dimgray')
                    plt.plot(x_coord_attn, y_coord_attn, 'o', color='red')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('Lettre: ' + str(idx) + ' Pred: ' + output[idx] + ' Target: ' + target[idx])
                
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
                plt.show()                
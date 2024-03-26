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
    trainning = False           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    withAtt = True          # Utiliser l'attention?
    bidirectionnal = False  # Utiliser un RNN bidirectionnel?

    # À compléter
    batch_size = 100            # Taille des lots
    n_epochs = 200              # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur

    n_hidden = 18               # Nombre de neurones caches par couches
    n_layers = 2               # Nombre de de couches

    os.makedirs('test_images', exist_ok=True)
    os.makedirs('test_images/'+ ('withAtt_' if withAtt else 'withoutAtt_')+ ('bidirectionnal' if bidirectionnal else 'unidirectionnal'), exist_ok=True)
    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    print(torch.cuda.is_available())
    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    # device = torch.device("cuda")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')

    # Séparation de l'ensemble de données (entraînement et validation)
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),
                                                                 int(len(dataset)*0.1), int(len(dataset)*0.1)])
    

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0)
    dataload_val = DataLoader(dataset_val,batch_size=batch_size,shuffle=True,num_workers=0)
    dataload_test = DataLoader(dataset_test,batch_size=batch_size,shuffle=False,num_workers=0)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('validation data : ', len(dataset_val))
    print('\n')

    # Instanciation du model
    model = trajectory2seq(n_hidden,n_layers,dataset.int2symb,dataset.symb2int,dataset.dictSize,device,dataset.maxLen, withAttention=withAtt, bidirectionnal=bidirectionnal)
    model = model.to(device)
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

    best_validation = -1 
    # Fonction de coût et optimizateur
    criterion = nn.CrossEntropyLoss(ignore_index=2) # ignorer les symboles <pad>
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if trainning:

        if learning_curves:
            val_loss =[] # Historique des distances
            train_loss=[] # Historique des coûts
            val_dist = [] # Historique des distances
            train_dist = [] # Historique des distances
            fig, (ax1, ax2) = plt.subplots(1, 2)


        for epoch in range(1, n_epochs + 1):
            running_loss_train = 0
            distTain=0
            model.train()
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
                    distTain += edit_distance(a[:Ma],b[:Mb])/batch_size


                # Affichage pendant l'entraînement
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    distTain/len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    distTain/len(dataload_train)), end='\r\n')

            # Validation
            running_loss_val = 0
            distVal=0
            model.eval()
            for batch_idx, data in enumerate(dataload_val):
                in_seq, target_seq = [obj.to(device).float() for obj in data]

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
                    distVal+= edit_distance(a[:Ma],b[:Mb])/batch_size


                # Affichage pendant l'entraînement
                print('Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_val.dataset),
                    100. * batch_idx *  batch_size / len(dataload_val.dataset), running_loss_val / (batch_idx + 1),
                    distVal/len(dataload_val)), end='\r')

            print('\n')
            if(distVal/len(dataload_val) < best_validation) or best_validation < 0:
                best_validation = distVal/len(dataload_val)
                print('Saving new best model\n')
                torch.save(model, 'model.pt')

            # Affichage graphique
            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                val_loss.append(running_loss_val/len(dataload_val))
                ax1.cla()
                ax1.plot(train_loss, label='training loss')
                ax1.plot(val_loss, label='validation loss')
                ax1.legend()

                train_dist.append(distTain/len(dataload_train))
                val_dist.append(distVal/len(dataload_val))
                ax2.cla()
                ax2.plot(train_dist, label='training dist')
                ax2.plot(val_dist, label='validation dist')
                ax2.legend()

                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            # torch.save(model,'model.pt')
        print("best validation: ", best_validation)

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.savefig('test_images/'+ ('withAtt_' if withAtt else 'withoutAtt_')+ ('bidirectionnal/' if bidirectionnal else 'unidirectionnal/') + 'learningCurve' + '.png')
            plt.show()
            plt.close('all')


 

    if test:
        # Évaluation
        model = torch.load('model.pt')
        model.device = device
        model = model.to(device)
        model.eval()
        dataset_test.symb2int = model.symb2int
        dataset_test.int2symb = model.int2symb
        
        seq_list = []   
        target_list_global = []
        output_list_global = []
        att_list = []

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
            
            seq_list.append(seq[:])
            target_list_global.append(target_list)
            output_list_global.append(output_list)
            att_list.append(att)
           
        # Affichage des résultats de test
        print('\nTest - Average loss: {:.4f} Average Edit Distance: {:.4f}'.format(running_loss_test/len(dataload_test), dist_test/len(dataload_test)))
        print('')   
        

        #add symbols to the lists
        for i in range(len(output_list)):
            output_list_int += [i for i in output_list[i]]
            target_list_int += [i for i in target_list[i]]
        ingore = [0,1,2]   
        matrix = confusion_matrix(target_list_int, output_list_int, ignore=ingore)
        plt.imshow(matrix)#,  cmap='binary')
        plt.xticks(np.arange(len(dataset.symb2int)-len(ingore)), [dataset.int2symb[i] for i in range(len(dataset.symb2int.keys())) if i not in ingore])
        plt.yticks(np.arange(len(dataset.symb2int)-len(ingore)), [dataset.int2symb[i] for i in range(len(dataset.symb2int.keys())) if i not in ingore])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('test_images/'+ ('withAtt_' if withAtt else 'withoutAtt_')+ ('bidirectionnal/' if bidirectionnal else 'unidirectionnal/') + 'confusionMatrix' + '.png')
        plt.show()
                    
        if gen_test_images:

            for i in range(10):
                random_batch = np.random.randint(0, len(output_list_global))
                random_idx = np.random.randint(0, len(output_list_global[random_batch]))
                
                seq = seq_list[random_batch][random_idx]
                output = output_list_global[random_batch][random_idx]
                target = target_list_global[random_batch][random_idx]
                att = att_list[random_batch][random_idx]
                
                # output = torch.argmax(torch.tensor(output), dim=1).detach().cpu().tolist()
                output = [model.int2symb[i] for i in output]
                
                # target = target.detach().cpu().tolist()
                target = [model.int2symb[i] for i in target]
                
                seq = seq.detach().cpu().tolist()
                
                att = att.detach().cpu().numpy()
                
                x_dist = []
                y_dist = []
                x_coord = np.zeros(1)
                y_coord = np.zeros(1)
                
                for idy in range(len(seq)-1):
                    x_dist.append(seq[idy][0])
                    y_dist.append(seq[idy][1])                
                    x_coord = np.append(x_coord, x_coord[-1] + x_dist[idy])
                    y_coord = np.append(y_coord, y_coord[-1] + y_dist[idy])
                    
                for idx in range(len(att)):
                    # x_coord_attn = x_coord[np.argpartition(att[idx], -100)[-100:]]
                    # y_coord_attn = y_coord[np.argpartition(att[idx], -100)[-100:]]
                    plt.subplot(3,2, idx+1)
                    plt.scatter(x_coord, y_coord, c=np.array(att[idx]), cmap='gray_r',norm='log', s=10)
                    # plt.plot(x_coord_attn, y_coord_attn, 'o', color='red')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('Lettre ' + str(idx) + ' Pred: ' + output[idx] + ' Target: ' + target[idx])
                
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
                plt.savefig('test_images/'+ ('withAtt_' if withAtt else 'withoutAtt_')+ ('bidirectionnal/' if bidirectionnal else 'unidirectionnal/') + 'image_' + str(i) + '.png', dpi=300)
                # plt.show()
                
                pass
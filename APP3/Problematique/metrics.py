# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition
    dist_matrix = np.zeros((len(x) + 1, len(y) + 1))

    for i in range(len(x) + 1):
        dist_matrix[i][0] = i

    for j in range(len(y) + 1):
        dist_matrix[0][j] = j

    v1 = 0 
    v2 = 0
    v3 = 0

    for t1 in range(1, len(x) + 1):
        for t2 in range(1, len(y) + 1):
            if (x[t1 - 1] == y[t2 - 1]):
                dist_matrix[t1][t2] = dist_matrix[t1-1][t2-1]
            else:
                v1 = dist_matrix[t1][t2-1]
                v2 = dist_matrix[t1-1][t2]
                v3 = dist_matrix[t1-1][t2-1]
            
                if (v1 <= v2 and v1 <= v3):
                    dist_matrix[t1][t2] = v1 + 1
                elif (v2 <= v1 and v2 <= v3):
                    dist_matrix[t1][t2] = v2 + 1
                else:
                    dist_matrix[t1][t2] = v3 + 1

    return dist_matrix[len(x)][len(y)]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    
    return None

import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    #Initialisation de la matrice
    dist_matrix = np.zeros((len(a) + 1, len(b) + 1))

    for i in range(len(a) + 1):
        dist_matrix[i][0] = i

    for j in range(len(b) + 1):
        dist_matrix[0][j] = j

    v1 = 0 
    v2 = 0
    v3 = 0

    for t1 in range(1, len(a) + 1):
        for t2 in range(1, len(b) + 1):
            if (a[t1 - 1] == b[t2 - 1]):
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

    # print(dist_matrix)

    return dist_matrix[len(a)][len(b)]
       
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    
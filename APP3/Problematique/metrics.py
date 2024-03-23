# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

# def edit_distance(x,y):
#     # Calcul de la distance d'édition
#     dist_matrix = np.zeros((len(x) + 1, len(y) + 1))

#     for i in range(len(x) + 1):
#         dist_matrix[i][0] = i

#     for j in range(len(y) + 1):
#         dist_matrix[0][j] = j

#     v1 = 0 
#     v2 = 0
#     v3 = 0

#     for t1 in range(1, len(x) + 1):
#         for t2 in range(1, len(y) + 1):
#             if (x[t1 - 1] == y[t2 - 1]):
#                 dist_matrix[t1][t2] = dist_matrix[t1-1][t2-1]
#             else:
#                 v1 = dist_matrix[t1][t2-1]
#                 v2 = dist_matrix[t1-1][t2]
#                 v3 = dist_matrix[t1-1][t2-1]
            
#                 if (v1 <= v2 and v1 <= v3):
#                     dist_matrix[t1][t2] = v1 + 1
#                 elif (v2 <= v1 and v2 <= v3):
#                     dist_matrix[t1][t2] = v2 + 1
#                 else:
#                     dist_matrix[t1][t2] = v3 + 1

#     return dist_matrix[len(x)][len(y)]

def edit_distance(x,y):
    # Calcul de la distance d'édition
    m = len(x)
    n = len(y)
    
    # Create a matrix to store the distances
    dp = np.zeros((m+1, n+1))
    
    # Initialize the first row and column
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Calculate the distances
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    
    return None

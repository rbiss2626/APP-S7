# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

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
    matrix = np.zeros((29-len(ignore),29-len(ignore)))
    if len(true) != len(pred):
        raise ValueError("Les deux listes doivent être de la même longueur")
    for i in range(len(true)):
        if true[i] not in ignore and pred[i] not in ignore:
            matrix[true[i]-len(ignore)][pred[i]-len(ignore)] += 1
            
    return matrix

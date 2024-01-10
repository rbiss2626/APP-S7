import numpy as np
import matplotlib.pyplot as plt

# A = np.array([[3,4,1], [5,2,3], [6,2,2]])
# A = np.array([[3,4,1,2,1,5], [5,2,3,2,2,1], [6,2,2,6,4,5], [1,2,1,3,1,2], [1,5,2,3,3,3], [1,2,2,4,2,1]])
A = np.array([[2,1,1,2], [1,2,3,2], [2,1,1,2], [3,1,4,1]])
B = np.random.uniform(0,1, size=A.shape)
I = np.eye(4)

mu = 0.002
nb_iteration = 10000

L_array = []

for i in range(nb_iteration):
    grad = 2*(B @ A - I) @ A.T
    L = np.sum((B @ A - I) ** 2)
    B = B - mu*grad
    L_array.append(L)

print (np.round(B, decimals=3))
print (np.round(B @ A, decimals=3))

x = range(0, nb_iteration)
plt.scatter(x, L_array)
plt.show()

#Problème 1 
# Avec des pas trop grands, on saute trop rapidement et on vient manquer le minimum. On s'emballe et s'éloigne. La solution est instable
# Le pas de 0.001 est trop petit, on pourrait arriver a un solution avec plus d'itérations, mais le pas de 0.005 est plus rapide et maintient
# la stabilité

#Probleme 2
# Il suffit de changer les hyperparametres pour etre en mesure de trouver une solution. Fonctionne bien avec un pas de 0.002 et 100 000 iterations

#Probleme 3
# La matrice est non inversible. On est quand meme capable de trouver un solution, mais la fonction de cout converge a 1 au lieu de 0. On se rend
# le plus proche possible d'une solution optimale sans pouvoir y arriver parfaitement. 
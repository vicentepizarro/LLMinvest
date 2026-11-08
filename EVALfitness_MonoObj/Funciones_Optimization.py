import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score, RepeatedStratifiedKFold
from math import inf

   
# Clasificador Regresión Logística para el cálculo del costo
# Se define cv fuera para asegurar consistencia de los modelos creados durante el proceso de optimización
def fitness_function_RL(curr_sol_bin, X_train, y_train, optimization,args):  # optimization = min ó max
    X_subset = X_train[:,curr_sol_bin]  #Conjunto con las features seleccionadas
    if X_subset.shape[1] == 0:
        print(f"Error. o puede evaluar un subconjunto vacío: {curr_sol_bin}")
        return float('inf')  # error alto
    regLogist = LogisticRegression(max_iter=args.max_iterations_fitness) # se crea una instancia nueva para entrenar y evaluar este nuevo modelo
    calidadades = cross_val_score(regLogist, X_subset, y_train, cv=args.cv_fitness, scoring='roc_auc')
    mean_calidad = np.mean(calidadades)
    if optimization == 'min':
        return  (1 - mean_calidad)
    else:           
        return mean_calidad    
      
    
import optuna
import numpy as np
import matplotlib.pyplot as plt
from Operadores.Operadores_solUnica import generar_solucion_ini, calcular_costo_regLogist, local_search_E1, roulette_wheel, apply_operador, run_operador
from Operadores.Operadores_solUnica import  filtrar_correlaciones, promedios_movimiento_costo, vecindarioVNS, vecindarioVNS_sinLS, local_search_sinLS
from tools_mh import input_output_data as io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
import logging
from math import inf

# Definir variable global para args
glob_args = None
def set_args(args):
    global glob_args
    glob_args = args

   
#--
#  ########################################################################
#
#                           METAHEURISTICA:       VARIABLE NEIGHBORHOOD SEARCH (VNS)
#
#  ########################################################################

def variable_neighborhood_search(features=None, dataX=None, targetY=None, trial=None):

    # Dividir Datos
    #X_train,X_test,y_train,y_test = train_test_split(dataX, targetY, test_size = glob_args.test_size, #0.10
    #                                                                                    random_state = glob_args.seed,  stratify=targetY)
    # Obtener correlación de X_train y filtrar según umbral
    corr_matriz = np.corrcoef(dataX, rowvar=False)
    corr_matriz = filtrar_correlaciones(corr_matriz)
    
    # Obtener genes altamente correlacionados en una lista: ==> sirve para obtener el complemento
    genes_bin = np.zeros(features.shape[0], dtype=bool) # arreglo binario con todos los genes
    genes_alta_corr = np.unique(np.where(corr_matriz > 0)[0])
    genes_bin[genes_alta_corr] = True # asigna True a los genes con alta corr
    
    # Obtener genes que no estén en la estructura de alta correlación
    genes_baja_corr = np.where(~genes_bin)[0]   # lo mismo np.where(genes_bin==False)[0]
    
    # Escalar datos para usar clasificadores como evaluadores de los subconjuntos de FS 
    scaler = StandardScaler()
    dataX_scaled = scaler.fit_transform(dataX)

    # Definir cómo evaluar el clasificador  (explicación al inicio) a través del parámetro cv de cross_val_score(...)
    #rkf = RepeatedStratifiedKFold(n_splits=glob_args.k_fold, n_repeats=glob_args.repeats_cv, random_state=glob_args.seed)
    random_state = np.random.randint(1,1000000)
    rkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)       
        
    # Generar Solución Inicial y evaluarla:
    num_evaluaciones = 0
    curr_sol_bin =  generar_solucion_ini(dataX.shape[1]) 
    curr_costo = calcular_costo_regLogist(curr_sol_bin, dataX_scaled, targetY, rkf)
    best_sol,  best_costo = curr_sol_bin, curr_costo 
    num_evaluaciones = num_evaluaciones + 1 
    
    # Comeanzar con las iteraciones hasta completar args.num_eval_max evaluaciones
    num_features_inicial = np.sum(curr_sol_bin)
    curr_costo_inicial = curr_costo
    
    # Parámetros de la vecindad
    k_max = vecindarioVNS.shape[0]  # Número máximo de vecindarios para shaking   

    # Criterio de detención: cuando se cumpla   el número de evaluaciones máxima
    while num_evaluaciones < glob_args.num_eval_max: 
    
        vecindario = 0
        while  vecindario < k_max:  #La últimas tres agitaciones son más intensas
            # ----------------------------------------------------------------------------------------------------------
            #  CON LS de correlación
            # Shaking: Generar solución aleatoria del k-ésimo vecindario de x (curr_sol) y evaluar
            #sol_shak = apply_operador(vecindarioVNS[vecindario],curr_sol_bin, corr_matriz,genes_baja_corr)
            #costo_sol_shak  = calcular_costo_regLogist(sol_shak, dataX_scaled, targetY, rkf)
            #num_evaluaciones = num_evaluaciones + 1
            #sol_LS, costo_sol_LS, num_evaluaciones = local_search_E1(sol_shak, costo_sol_shak, corr_matriz, genes_baja_corr, num_evaluaciones, dataX_scaled, targetY, rkf)   
            # -------------------------------------------------------------------------------------
            
            
            # ----------------------------------------------------------------------------------------------------------
            #  SIN LS: sin correlación
            sol_shak = apply_operador(vecindarioVNS_sinLS[vecindario],curr_sol_bin, corr_matriz,genes_baja_corr)
            costo_sol_shak  = calcular_costo_regLogist(sol_shak, dataX_scaled, targetY, rkf)
            num_evaluaciones = num_evaluaciones + 1
            sol_LS, costo_sol_LS, num_evaluaciones = local_search_sinLS(sol_shak, costo_sol_shak, num_evaluaciones, dataX_scaled, targetY, rkf)   
            # ----------------------------------------------------------------------------------------------------------
                        
            # Revisar si la búsqueda se mueve al siguiente vecindario (esto es cuando curr_costo no mejora)
            if  costo_sol_LS < curr_costo:
                curr_sol_bin, curr_costo = sol_LS, costo_sol_LS  # nuevo optimo local
                best_sol,  best_costo = sol_LS.copy(), costo_sol_LS.copy()
                vecindario = 0  #Reinicia la búqueda desde el vecindario 0
            else:  
                vecindario = vecindario + 1 # La búsqueda se mueve al vecindario siguiente (cambia de entorno para probocar una perturbación mayor)

            # .......................................................................................................................................
            # Reportar  valor intermedio para optuna y compueba si ensayo debe ser podado
            trial.report(curr_costo, num_evaluaciones)  # mejor óptimo local encontrado hasta ahora
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # -----------------------------------------------------------------------------------

            # Verificar criterio de detención 
            if  num_evaluaciones > glob_args.num_eval_max:
                break # salir del recorrido de los vecindarios
            
    return features[best_sol], best_costo  #np.where(best_solution)[0], best_costo
    
    
    

import optuna
import numpy as np
import matplotlib.pyplot as plt
from EVALfitness_MonoObj.Funciones_Optimization import  fitness_function_RL
from Operadores.Operadores_solUnica import generar_solucion_ini, local_search_E1, local_search_sinLS, filtrar_correlaciones
from Operadores.Operadores_solUnica import vecindarioVNS_sinLS, vecindarioVNS_conLS, apply_operador_2, apply_operador
from sklearn.preprocessing import StandardScaler
import logging
from math import inf

   
#--
#  ########################################################################
#
#               METAHEURISTICA: VARIABLE NEIGHBORHOOD SEARCH (VNS)
#               Minimiza pero se informa el 1-costo
#  ########################################################################

def variable_neighborhood_search(features=None, dataX=None, targetY=None, args=None, trial=None):
    
    # Obtener correlación de X_train y filtrar según umbral
    corr_matriz = np.corrcoef(dataX, rowvar=False)
    corr_matriz = filtrar_correlaciones(corr_matriz,args.umbral_correlacion)    
    
    # Obtener genes altamente correlacionados en una lista: ==> sirve para obtener el complemento
    genes_bin = np.zeros(features.shape[0], dtype=bool) # arreglo binario con todos los genes
    genes_alta_corr = np.unique(np.where(corr_matriz > 0)[0])
    genes_bin[genes_alta_corr] = True # asigna True a los genes con alta corr

    # Obtener genes que no estén en la estructura de alta correlación
    genes_baja_corr = np.where(~genes_bin)[0]   # lo mismo np.where(genes_bin==False)[0]

    # Escalar datos para usar clasificadores como evaluadores de los subconjuntos de FS 
    scaler = StandardScaler()
    dataX_scaled = scaler.fit_transform(dataX)

    # Generar Solución Inicial y evaluarla:
    num_evaluaciones = 0
    curr_sol_bin =  generar_solucion_ini(dataX.shape[1], args.gene_initial_prob) 
    curr_costo = fitness_function_RL(curr_sol_bin, dataX_scaled, targetY, 'min', args)
    best_sol,  best_costo = curr_sol_bin, curr_costo 
    num_evaluaciones = num_evaluaciones + 1 
    
    if args.isOptuna == 'no':
        logging.info(f"Solucion (evaluaciones, fitness, cantidad_FS)\n{num_evaluaciones},{1-curr_costo:.6f},{np.sum(curr_sol_bin)}")
    
    # Criterio de detención: cuando se cumpla   el número de evaluaciones máxima
    while num_evaluaciones < args.num_eval_max: 
           
        # Recorrer los vecindarios
        vecindario = 0
        while  vecindario < len(vecindarioVNS_conLS):  #La últimas tres agitaciones son más intensas
                    
            if args.isLS == 'no':
                sol_shak = apply_operador(vecindarioVNS_sinLS[vecindario],curr_sol_bin)
                costo_sol_shak  = fitness_function_RL(sol_shak, dataX_scaled, targetY, 'min', args) 
                num_evaluaciones = num_evaluaciones + 1
                sol_LS, costo_sol_LS, num_evaluaciones = local_search_sinLS(sol_shak, costo_sol_shak, num_evaluaciones, dataX_scaled, targetY, args)   
            else:
                # ----------------------------------------------------------------------------------------------------------
                #  CON LS (de correlación)
                # Shaking: Generar solución aleatoria del k-ésimo vecindario de x (curr_sol) y evaluar
                sol_shak = apply_operador_2(vecindarioVNS_conLS[vecindario],curr_sol_bin, genes_baja_corr)
                costo_sol_shak  = fitness_function_RL(sol_shak, dataX_scaled, targetY, 'min', args) 
                num_evaluaciones = num_evaluaciones + 1
                sol_LS, costo_sol_LS, num_evaluaciones = local_search_E1(sol_shak, costo_sol_shak, corr_matriz, num_evaluaciones, dataX_scaled, targetY, args)   
            if args.isOptuna=='no':
                logging.info(f"{num_evaluaciones},{1-costo_sol_LS:.6f},{np.sum(sol_LS)}")
                                  
            # Revisar si la búsqueda se mueve al siguiente vecindario (esto es cuando curr_costo no mejora)
            if  costo_sol_LS < curr_costo:
                curr_sol_bin, curr_costo = sol_LS, costo_sol_LS  # nuevo optimo local
                best_sol,  best_costo = sol_LS.copy(), costo_sol_LS.copy()
                vecindario = 0  #Reinicia la búqueda desde el vecindario 0
            else:  
                vecindario = vecindario + 1 # La búsqueda se mueve al vecindario siguiente (cambia de entorno para probocar una perturbación mayor)

            # .......................................................................................................................................
            # Reportar  valor intermedio para optuna y compueba si ensayo debe ser podado
            if args.isOptuna=='si':
                trial.report(1-curr_costo, num_evaluaciones)  # mejor costo encontrado hasta ahora
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            # -----------------------------------------------------------------------------------

            # Verificar criterio de detención 
            if  num_evaluaciones >= args.num_eval_max:
                break # salir del recorrido de los vecindarios
            
    if args.isOptuna=='no':
        selected_genes_indices = np.where(best_sol)[0]
        selected_genes = ','.join(features[selected_genes_indices].tolist())
        logging.info(f"Genes de la mejor solución SA:\n{selected_genes}")
        
    return best_sol, 1-best_costo    
    
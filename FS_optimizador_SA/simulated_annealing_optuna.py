import optuna
import numpy as np
import matplotlib.pyplot as plt
from EVALfitness_MonoObj.Funciones_Optimization import  fitness_function_RL
from Operadores.Operadores_solUnica import generar_solucion_ini, local_search_E1, local_search_sinLS
from Operadores.Operadores_solUnica import run_operador_conLS, run_operador_sinLS, filtrar_correlaciones, promedios_movimiento_costo
#from FS_optimizador_SA import config_param as param
from tools_mh import input_output_data as io
from sklearn.preprocessing import StandardScaler
from math import inf


#--
#  ########################################################################
#
#           METAHEURISTICA: SIMULATED ANNEALING PARA SELECCIÓN DE CARACTERÍSTICAS (FS)
#
#  ########################################################################

def simulated_annealing(features, dataX, targetY, trial, args):
     
    # Obtener correlación de X_train y filtrar según umbral
    corr_matriz = np.corrcoef(dataX, rowvar=False)
    corr_matriz = filtrar_correlaciones(corr_matriz, args.umbral_correlacion)    
    
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
    
    # Comeanzar con las iteraciones hasta completar args.num_eval_max evaluaciones
    num_features_inicial = np.sum(curr_sol_bin)
    curr_costo_inicial = curr_costo
    temperature = args.ini_temp
     
    # Criterio de detención: cuando se cumpla   el número de evaluaciones máxima
    while num_evaluaciones < args.num_eval_max: 
    
        # Recorrer SS para cada temperatura
        for iteracion in range(1, args.repeats_temp): # número de vecinos por temperatura
        
            if args.isLS == 'no':
                # -------------------------------------------------------------------------------------------
                # SIN LS     
                new_sol = run_operador_sinLS(curr_sol_bin) 
                new_costo = fitness_function_RL(new_sol, dataX_scaled, targetY, 'min', args)                
                num_evaluaciones = num_evaluaciones + 1           
                new_sol_LS, new_costo_LS, num_evaluaciones = local_search_sinLS(new_sol, new_costo, num_evaluaciones, dataX_scaled, targetY, args)  
                # -------------------------------------------------------------------------------------------
            else:
                # -------------------------------------------------------------------------------------------
                # CON LS     
                new_sol = run_operador_conLS(curr_sol_bin, genes_baja_corr)  
                new_costo = fitness_function_RL(new_sol, dataX_scaled, targetY, 'min', args)                
                num_evaluaciones = num_evaluaciones + 1
                new_sol_LS, new_costo_LS, num_evaluaciones = local_search_E1(new_sol, new_costo, corr_matriz, num_evaluaciones, dataX_scaled, targetY, args)                  
            # -------------------------------------------------------------------------------------------
            
            # Para ambos casos
            if new_costo_LS < new_costo:
               new_sol, new_costo = new_sol_LS, new_costo_LS
            
            # ACEPTAR O RECHAZAR nueva solución según SA
            dif = new_costo - curr_costo # dif negativo: new_costo es mejor que curr_costo
            if dif < 0: # Siempre se acepta una solución de mejor costo
                curr_sol_bin, curr_costo = new_sol, new_costo
                if curr_costo < best_costo:
                    best_sol,  best_costo = curr_sol_bin.copy(), curr_costo.copy()
                    num_eval_mejora = num_evaluaciones
                    
            else: # Si no es mejor. Prodría o no aceptarla
            
                prob = np.exp(-dif/temperature)
                if  prob > np.random.rand(): #Se acepta
                    curr_sol_bin, curr_costo = new_sol, new_costo

            # .......................................................................................................................................
            # Reportar  valor intermedio para optuna y compueba si ensayo debe ser podado
            trial.report(curr_costo, num_evaluaciones)  # mejor costo encontrado hasta ahora
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # -----------------------------------------------------------------------------------
            
            # Verificar criterio de detención 
            if  num_evaluaciones >= args.num_eval_max:
                break # salir del enfriamiento de la temp
                
        # Modificar Temperatura
        temperature = args.alpha_func_cool*temperature
        
        # Verificar temperatura alcanzada 
        if  temperature < args.min_temp: #0.001
            break
                       
    return features[best_sol], best_costo  #np.where(best_solution)[0], best_costo



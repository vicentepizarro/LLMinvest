import optuna
from Operadores.Operadores_solPoblacion import create_initial_population, selection, crossover, mutation
from Operadores.Operadores_solUnica import  filtrar_correlaciones, local_search_E1
from EVALfitness_MonoObj.Funciones_Optimization import  fitness_function_RL
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import Parallel, delayed # facilita la paralelización de tareas
import logging

def AG(features_filter=None, data=None, targetY=None, args=None, trial=None):

    # Obtener matriz de correlación y filtrar para umbral. (no se escalan los datos para obtener las correlaciones)
    corr_matriz = np.corrcoef(data, rowvar=False)
    corr_matriz = filtrar_correlaciones(corr_matriz, args.umbral_correlacion )    
    
    # Escalar datos para usar modelo RL como evaluador de los subconjuntos de FS 
    scaler = StandardScaler()
    dataX_scaled = scaler.fit_transform(data)

    num_evaluaciones = 0
    
    # Generar matriz de pobración inicial: dimension (pop_size, num_genes)
    pop_parents=create_initial_population(args.pop_size, dataX_scaled.shape[1], args.gene_initial_prob) 
    
    # Evaluar toda la población inicial. Usaremos Paralelismo: Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in iterable)
    fitness_results = np.array(Parallel(n_jobs=args.n_jobs_fitness)(delayed(fitness_function_RL)(sol, dataX_scaled, targetY, 'max',args) for sol in pop_parents) )
    num_evaluaciones = num_evaluaciones + pop_parents.shape[0]
    
    sorted_indices = np.argsort(fitness_results)[::-1]
    fitness_sorted_indices = fitness_results[sorted_indices]
    population_sorted_indices = pop_parents[sorted_indices]
    if args.isOptuna=='no':
        logging.info(f"POBLACION con mejor solucion (evaluaciones, fitness, cantidad_FS)\n{num_evaluaciones},{fitness_sorted_indices[0]:.6f},{np.sum(population_sorted_indices[0])}")
       
    while num_evaluaciones < args.num_eval_max:
    
        # SELECTION de genes para el cruce. Se crean dos grupos:  pop_initial (elitista) y pop_remaining (no elitista).  Ambos suman el tamaño de la poblacion
        pop_remaining, pop_elitist, fitness_pop_elitist  = selection(pop_parents, fitness_results, args.percentage_select)  

        # CRUCE  (hay controlar que tamaño pop_children = pop_remaining)
        pop_children = crossover(pop_remaining, pop_elitist, args.crossover_rate)  

        #MUTAR: ¿sólo se puta una parte de la nueva población? (se deja fuera pop_initial)
        pop_children = mutation(pop_children, args.mutation_rate, args.gene_mutation_rate) 

        # Evaluar soluciones mutadas (pop_children)
        fitness_pop_children = np.array(Parallel(n_jobs=args.n_jobs_fitness)(delayed(fitness_function_RL)(sol, dataX_scaled, targetY, 'max',args) for sol in pop_children) )
        num_evaluaciones = num_evaluaciones + pop_children.shape[0]

        #
        #  BÚSQUEDA LOCAL
        #
        if args.isLS == 'si':
            # Para los descendientes pop_children se mejora  fitness con búsqueda local (LS)
            for indiv  in range(pop_children.shape[0]):
                pop_children[indiv], new_costo_LS, num_evaluaciones = local_search_E1(pop_children[indiv], 1 - fitness_pop_children[indiv], corr_matriz, num_evaluaciones, dataX_scaled, targetY, args)  
                new_fitness = 1 - new_costo_LS
                fitness_pop_children[indiv] = new_fitness
            
        # Combinar padres y descendientes para generar la nueva población pop_parents
        pop_parents = np.vstack((pop_elitist, pop_children))
        fitness_results = np.concatenate((fitness_pop_elitist, fitness_pop_children))
        
        # Ordenar Fitness de Mayor  a menor  por fitness
        sorted_indices = np.argsort(fitness_results)[::-1]
        fitness_sorted_indices = fitness_results[sorted_indices]
        pop_parents_sorted = pop_parents[sorted_indices]
        if args.isOptuna=='no':
            logging.info(f"{num_evaluaciones},{fitness_sorted_indices[0]:.6f},{np.sum(pop_parents_sorted[0])}")

        # .......................................................................................................................................
        # Reportar  valor intermedio para optuna y comprueba si ensayo debe ser podado
        if args.isOptuna=='si':
            trial.report(fitness_sorted_indices[0], num_evaluaciones)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        # .......................................................................................................................................

        # Verificar criterio de detención 
        if  num_evaluaciones >= args.num_eval_max:
            break
    if args.isOptuna=='no':
        selected_genes_indices = np.where(pop_parents_sorted[0])[0]
        selected_genes = ','.join(features_filter[selected_genes_indices].tolist())
        logging.info(f"Genes con mejor solución:\n{selected_genes}")
    return  pop_parents_sorted[0], fitness_sorted_indices[0]   # Envía la población final y fitness ordenados de mayor a menor
  
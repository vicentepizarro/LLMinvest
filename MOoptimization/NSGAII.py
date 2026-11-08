#__________________________________________________
#
#       OPTIMIZADOR NSGA-II
#.-_______________________________________________

import numpy as np
from joblib import Parallel, delayed 
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.indicators.hv import HV
from MOoptimization.Operadores_solPoblacion import (selection, crossover, mutation, supervivencia, initial_pop)
from MOoptimization.MOmethods_segumiento import (log_full)
import optuna
import EVALfitness.eval as eval #referencia para usar cualquier cosa de eval.py
import LS.operadores_LS as lsops #referencia para usar cualquier cosa de operadores_LS

nds = NonDominatedSorting()

# --- HIPERVOLUMEN NORMALIZADO --------
class NormalizedHV:
    def __init__(self, n_genes):
        self.hv = HV(ref_point=np.array([1.000001, float(n_genes)])) #self.hv se configura para posteriormente calcular hv con self.hv.do
        self.hv_max = float(1.000001 * n_genes)
    def do(self, F):
        return float(self.hv.do(F)) / self.hv_max   # self.hv.do(F) calcula el hipervolumen real de F respecto a [1.000001, n_genes]
                                                    # la división por self.hv_max es para normalizar entre [0,1]
def inicializar_convergencia(n_genes):
    return NormalizedHV(n_genes) # sólo instancia un objeto definiendo self.hv y self.hv_max


# --- UTILIDADES DE SEGUIMIENTO -----
def row_ids_bool(pop_bool):
    packed = np.packbits(np.asarray(pop_bool, dtype=np.bool_), axis=1)
    return [row.tobytes() for row in packed]

def control_metrics(fitness, fronts, hv_tool):
    idxs_F0 = np.asarray(fronts[0], dtype=int)
    if idxs_F0.size:
        F0 = np.asarray(fitness[idxs_F0], dtype=float)
        F0[:, 0] = np.clip(F0[:, 0], 0.0, 1.0)
        hv0 = float(hv_tool.do(F0))
    else:
        hv0 = 0.0
    return idxs_F0, hv0

def update_no_improve(state, hv0): #Contador de generaciones sin mejorar HV (lo consume la LS)
    prev = state.get("prev_hv0", None)
    if prev is not None and np.isfinite(prev):
        state["count_no_improve_hv0"] = 0 if hv0 > prev + 1e-12 \
            else state["count_no_improve_hv0"] + 1
    else:
        state["count_no_improve_hv0"] = 0
    state["prev_hv0"] = hv0
    return state




#.- ===========================================

def NSGA2(G, X_no_scaled=None, targetY=None, comunidad_of_nodos=None, inds_comunidades=None, 
          top_nodes_por_comunidad=None, trial=None,*, args, splits, seed_clf, rng_global, rng_init, seed_ls): 
    
    # Construir clave de contexto del caché (única para este dataset/clf/splits).
    ctx_key = eval.context_key(args.nom_dataset, args.clasificador, seed_clf,  args.cv_folds)
    num_evaluaciones=0
    generacion = 0
    last_reported = 0 # para reportes en optuna
    state = {"prev_hv0": None, "count_no_improve_hv0": 0} #state informativo SÓLO PARA SEGUIMIENTO
    evals_LS = np.array([], dtype=int)
    diags = []
    dup_child = 0.0
    #------------------------------------
    # Adyacencia CSR para la LS: se transmite UNA vez por generación y se comparte
    # read-only entre workers. 
    ls_indptr = ls_indices = ls_P = None
    if args.metodo == "NSGA-II+mKNNG-LS" and G is not None:
        import networkx as nx  # requiere networkx >= 2.7
        n_nodes = X_no_scaled.shape[1]
        A = nx.to_scipy_sparse_array(G, nodelist=range(n_nodes), format="csr", weight=None)
        ls_indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        ls_indices = np.ascontiguousarray(A.indices, dtype=np.int32)    
        ls_P = lsops.participation_coefficient(ls_indptr, ls_indices, comunidad_of_nodos)
    #.----------------------------------
    hv_tool = inicializar_convergencia(X_no_scaled.shape[1])

    #.-
    #.- Operador (1): GENERAR POBLACIÓN INICIAL
    population = initial_pop(rng_init, args.pop_size, X_no_scaled.shape[1], args.gene_initial_prob)
    fitness_pop, evs_init = eval.eval_batch_with_cache_loky(population, X_no_scaled, targetY, splits, seed_clf, args.clasificador, 
                                                            n_jobs=args.n_jobs_fitness, ctx_key=ctx_key)
    num_evaluaciones += evs_init  
    fronts = nds.do(fitness_pop)
    _, hv0 = control_metrics(fitness_pop, fronts, hv_tool)
    state = update_no_improve(state, hv0)
    if args.isOptuna == 'no':
        log_full(generacion, num_evaluaciones, fitness_pop, hv0, state, dup_child, diags)

    while num_evaluaciones < args.num_evals_max:

        if args.isOptuna == 'si' and trial is not None and (num_evaluaciones - last_reported) >= args.interval_steps:
            running_mean = (args.completed_hv_sum + hv0) / (args.completed_count + 1)
            trial.report(running_mean, args.step_offset + num_evaluaciones)
            last_reported = num_evaluaciones
            if num_evaluaciones >= args.n_warmup_steps and trial.should_prune():
                    raise optuna.TrialPruned()
        
        #.-  Operadores (2), (3) y (4): SELECCION PADRES + CRUCE + MUTACIÓN
        remaining = args.num_evals_max - num_evaluaciones
        lam = min(args.pop_size, remaining)
        parents_a, parents_b = selection(rng_global, population, fitness_pop, fronts, lam)
        pop_children = crossover(rng_global, parents_a, parents_b, args.crossover_rate, lam)
        pop_children = mutation(rng_global, pop_children, args.mutation_rate, args.gene_mutation_rate, force_at_least_one_change=True) 
        dup_child = 1 - len(set(row_ids_bool(pop_children))) / pop_children.shape[0]
        
        #.- EVALUACIÓN hijos
        n_jobs = min(args.n_jobs_fitness, pop_children.shape[0])
        fitness_pop_children, evs_children = eval.eval_children_dedup(pop_children, X_no_scaled, targetY, splits, seed_clf, args.clasificador, n_jobs, ctx_key)
        num_evaluaciones += int(evs_children)

        #.- Operador (5): BÚSQUEDA LOCAL (sobre una fracción de pop_children)
        diags = [] # se resetea cada generación
        if args.metodo == "NSGA-II+mKNNG-LS":
            #.-
            pop_children, fitness_pop_children, evals_LS, diags = lsops.mkNNG_LS(args, pop_children, num_evaluaciones, fitness_pop_children, 
                                                                           seed_ls, generacion, ls_indptr, ls_indices, ls_P, X_no_scaled, 
                                                                           targetY, splits, seed_clf, inds_comunidades, comunidad_of_nodos, 
                                                                           top_nodes_por_comunidad, state["count_no_improve_hv0"], ctx_key,None,None)
            num_evaluaciones += int(evals_LS.sum())

        #.- Operador (6): SUPERVIVENCIA  
        population, fitness_pop, incluye_duplicados  = supervivencia(population, pop_children, fitness_pop, fitness_pop_children, nds)
        if incluye_duplicados and args.isOptuna == 'no':
            print(f"[WARN gen={generacion}] supervivencia incluye duplicados activado: "
                f"unión con <N únicos. Posible convergencia masiva." )
        generacion += 1

        fronts = nds.do(fitness_pop)
        _, hv0 = control_metrics(fitness_pop, fronts, hv_tool)
        state = update_no_improve(state, hv0)
        if args.isOptuna == 'no':
            log_full(generacion, num_evaluaciones, fitness_pop, hv0, state, dup_child, diags)


        if num_evaluaciones >= args.num_evals_max:
            break
    # END-WHILE
    #.

    # Deduplicar el frente
    packed = np.packbits(population.astype(np.bool_), axis=1) #convierte cada fila booleana population en una representación compacta por bytes.
    _, uniq_idx = np.unique(packed, axis=0, return_index=True)
    population_uniq = population[uniq_idx]
    fitness_pop_uniq = fitness_pop[uniq_idx]

    # Obtener F0 sin duplicados
    fronts = nds.do(fitness_pop_uniq)
    idxs = fronts[0]
    F0 = np.asarray(fitness_pop_uniq[idxs], dtype=float)
    F0[:, 0] = np.clip(F0[:, 0], 0.0, 1.0)
    F0_hv = float(hv_tool.do(F0))
    if trial is not None and last_reported != num_evaluaciones:
        running_mean_final = (args.completed_hv_sum + F0_hv) / (args.completed_count + 1)
        trial.report(running_mean_final, args.step_offset + num_evaluaciones)

    return population_uniq[idxs], fitness_pop_uniq[idxs], F0_hv



import numpy as np
from joblib import Parallel, delayed
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from MOoptimization.Operadores_solPoblacion import crossover, mutation, initial_pop
from MOoptimization.MOmethods_segumiento import (log_full)

import optuna
import EVALfitness.eval as eval
import LS.operadores_LS as lsops

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

# --- FUNCIONES PROPIAS DEL MOEA/D según paper: Multiobjective Evolutionary Algorithm Based on Decomposition
#     https://www.researchgate.net/publication/3418989_MOEAD_A_Multiobjective_Evolutionary_Algorithm_Based_on_Decomposition
#     Descomposición de Tchebycheff (Sección II-B del paper) ------------------
def pesos_uniformes(N):
    """N vectores de peso uniformes para m=2: λ^i = (i/(N-1), 1 - i/(N-1))."""
    a = np.linspace(0.0, 1.0, N)
    W = np.column_stack([a, 1.0 - a])  # (N, 2)
    return W

def vecindarios(W, T):
    """B(i) = los T subproblemas con λ más cercano (distancia euclidiana). Incluye i."""
    # distancias entre todos los pares de pesos
    diff = W[:, None, :] - W[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))        # (N, N)
    return np.argsort(D, axis=1)[:, :T]         # (N, T), la 1ª columna es i mismo

# fitnees multiobjetivo usando f1, f2 con sus respectivos pesos  por solución (o subproblema)
def tchebycheff(Fitness_row, pesos_j, z, z_nad):
    """g^te normalizado: max_j λ_j * |f_j - z_j| / (z_nad_j - z_j).
    La normalización (Sec. VII del paper) es indispensable aquí porque f1∈[0,1]
    y f2 puede valer miles: sin ella el max lo domina siempre f2."""
    denom = z_nad - z
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return float(np.max(pesos_j * np.abs(Fitness_row - z) / denom))

def tchebycheff_vecinos(fitness, vecinos, W, z, z_nad):
    denom = z_nad - z
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return np.max(W[vecinos] * np.abs(fitness - z) / denom, axis=1)

# --- ARCHIVO EXTERNO (EP, Step 2.5 del paper MOEA/D) ---------------------------
# Mantiene TODOS los no-dominados hallados durante la corrida (dedup por genotipo).
# Recupera soluciones compactas que luego son sobrescritas en su slot vecinal.
def _frente_no_dominado(pop_bool, fit):
    packed = np.packbits(pop_bool.astype(np.bool_, copy=False), axis=1)
    _, uniq = np.unique(packed, axis=0, return_index=True)
    pop_u, fit_u = pop_bool[uniq], fit[uniq]
    f0 = nds.do(fit_u)[0]
    return pop_u[f0], fit_u[f0]

def actualizar_archivo(arch_pop, arch_fit, new_pop, new_fit):
    if new_pop is None or len(new_pop) == 0:
        return arch_pop, arch_fit
    new_pop = np.asarray(new_pop, dtype=bool)
    new_fit = np.asarray(new_fit, dtype=float)
    if arch_pop is None:
        return _frente_no_dominado(new_pop, new_fit)
    allp = np.vstack([arch_pop, new_pop])
    allf = np.vstack([arch_fit, new_fit])
    return _frente_no_dominado(allp, allf)

#. ________________________________________________
# 
#            MOEA/D
#. _____________________________________________________
def MOEAD(G, X_no_scaled=None, targetY=None, comunidad_of_nodos=None, inds_comunidades=None,
          top_nodes_por_comunidad=None, trial=None, *, args, splits, seed_clf,
          rng_global, rng_init, seed_ls):
    ctx_key = eval.context_key(args.nom_dataset, args.clasificador, seed_clf, args.cv_folds)
    num_evaluaciones = 0
    generacion = 0
    last_reported = 0
    state = {"prev_hv0": None, "count_no_improve_hv0": 0} #state informativo SÓLO PARA SEGUIMIENTO
    evals_LS = np.array([], dtype=int)
    dup_child = 0.0
    diags = []
    #------------------------------------
    # Adyacencia CSR para la LS: se transmite UNA vez por generación 
    ls_indptr = ls_indices = ls_P = None
    if args.metodo == "MOEAD+mKNNG-LS" and G is not None:
        import networkx as nx  # requiere networkx >= 2.7
        n_nodes = X_no_scaled.shape[1]
        A = nx.to_scipy_sparse_array(G, nodelist=range(n_nodes), format="csr", weight=None)
        ls_indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        ls_indices = np.ascontiguousarray(A.indices, dtype=np.int32)    
        ls_P = lsops.participation_coefficient(ls_indptr, ls_indices, comunidad_of_nodos)
    #.----------------------------------
    hv_tool = inicializar_convergencia(X_no_scaled.shape[1])

    # --- Step 1.2 paper MOEA/D: pesos y vecindarios ---
    W = pesos_uniformes(args.pop_size)
    B = vecindarios(W, args.moead_T)

    # --- Step 1.3 paper MOEA/D: población inicial (misma que NSGA-II) ---
    # ejemplo de population es una matriz args.pop_size x n_genes, donde n_genes = X_no_scaled.shape[1], por ejemplo 3594 para Prostate_GSE6919_U95B 
    #            fitness_pop es una matriz args.pop_size x n_objetivos, donde n_objetvos = 2  (error clasificacion  y num_genes_activos)
    #    population                 fitness_pop
    #  1 0 0 0 1 0 0 1 0 ....        0.23, 115
    #  0 0 0 1 1 0 0 1 0 ....        0.13, 90
    #  :                             :
    population = initial_pop(rng_init, args.pop_size, X_no_scaled.shape[1], args.gene_initial_prob)
    fitness_pop, evs_init = eval.eval_batch_with_cache_loky(population, X_no_scaled, targetY, splits, seed_clf, args.clasificador, 
                                                            n_jobs=args.n_jobs_fitness, ctx_key=ctx_key)
    num_evaluaciones += evs_init  
    fronts = nds.do(fitness_pop)
    
    # --- Step 1.4 paper MOEA/D: z (ideal) y estimado, para la normalización ---
    z = fitness_pop.min(axis=0).astype(float)          # el mejor por objetivo separado
    # FIX #1: z_nad[1] = #genes de la solución más grande VIVA (NO el total del dataset),
    # para que f2 quede en escala comparable a f1 en Tchebycheff (se re-actualiza por gen).
    z_nad = np.array([1.000001, max(float(fitness_pop[:, 1].max()), z[1] + 1.0)])     

    # EP (Step 2.5 del paper): archivo externo de no-dominados (dedup por genotipo)
    arch_pop, arch_fit = actualizar_archivo(None, None, population, fitness_pop)

    if args.isOptuna == 'no':
        # CLAVE de validación: z_nad[1] ≈ #genes_max_vivos → FIX activo;  z_nad[1] == n_genes_dataset → BUG
        print("Optimización MOEA/D con archivo externo EP")
        print(f"[INIT MOEAD] z_ideal=[err={z[0]:.6f}, genes={z[1]:.0f}]  "
              f"z_nad=[{z_nad[0]:.6f}, {z_nad[1]:.1f}]  n_genes_dataset={X_no_scaled.shape[1]}  "
              f"pop_size={args.pop_size} moead_T={args.moead_T}", flush=True)

    # Seguimiento generación 0
    _, hv0 = control_metrics(fitness_pop, fronts, hv_tool)
    state = update_no_improve(state, hv0)
    if args.isOptuna == 'no':
        log_full(generacion, num_evaluaciones, fitness_pop, hv0, state, dup_child, diags)



    # --- Step 2 / Step 3 paper MOEA/D: loop hasta agotar presupuesto ---
    while num_evaluaciones < args.num_evals_max:

        z_nad[1] = max(float(fitness_pop[:, 1].max()), z[1] + 1.0)  # FIX #1: nadir adaptativo

        if state["count_no_improve_hv0"] >= 10000:
            print(f"[STOP] Sin mejora de HV durante {state['count_no_improve_hv0']} generaciones consecutivas." )
            break        

        if args.isOptuna == 'si' and trial is not None and (num_evaluaciones - last_reported) >= args.interval_steps:
            running_mean = (args.completed_hv_sum + hv0) / (args.completed_count + 1)
            trial.report(running_mean, args.step_offset + num_evaluaciones)
            last_reported = num_evaluaciones
            if num_evaluaciones >= args.n_warmup_steps and trial.should_prune():
                raise optuna.TrialPruned()

        #if args.metodo == "MOEAD+mKNNG-LS":
            # ¿cada cuántos subproblemas aplico LS?. Si LS_frac_children= 0.13 => 1/0.13 = 7.7: Aquí aplicaré LS a 1 de cada 8 subproblemas aprox
        #    intervalo_LS = max(1, round(1.0 / max(args.LS_frac_children, 1e-9)))  # 1 de cada intervalo_LS

        #   RECORRER SUBPROBLEMAS EN ORDEN ALEATORIO (todos los subproblemas corresponden a una generación )
        # ------------------------------------------------
        hijos_gen, fit_gen = [], []   # EP: acumula lo evaluado en esta generación
        diags = []
        orden_recorrido_subprobs = rng_global.permutation(args.pop_size)
        for ini_batch in range(0, args.pop_size, args.moead_batch_size):
            if num_evaluaciones >= args.num_evals_max:
                break
            n_restantes = args.num_evals_max - num_evaluaciones
            batch_subprobs = orden_recorrido_subprobs[ini_batch:ini_batch + min(args.moead_batch_size, n_restantes)]
            if batch_subprobs.size  == 0:
                break

            tam_batch =  batch_subprobs.size
            hijos = np.empty((tam_batch, population.shape[1]), dtype=bool)
            meta_hijos = np.empty((tam_batch, 2), dtype=int)
            for k_local, num_subproblema in enumerate(batch_subprobs):
                k_esimo_subprob_recorrido = ini_batch + k_local
                k, l = rng_global.choice(B[num_subproblema], size=2, replace=False)
                pa = population[k][None, :]
                pb = population[l][None, :]
                hijo = crossover(rng_global, pa, pb, args.crossover_rate, lam=1)
                hijo = mutation(rng_global, hijo, p_mut_ind=1, p_mut_gene=args.gene_mutation_rate, force_at_least_one_change=True)
                hijos[k_local] = hijo[0]
                meta_hijos[k_local] = (k_esimo_subprob_recorrido, num_subproblema)
            #end-for
            n_jobs_batch = min(args.n_jobs_fitness, hijos.shape[0])
            fitness_hijos, evs_batch = eval.eval_children_dedup(hijos, X_no_scaled, targetY, splits, seed_clf, args.clasificador, 
                                                                        n_jobs=n_jobs_batch, ctx_key=ctx_key)
            num_evaluaciones += evs_batch
            #.-
            # Aprovechar estos procesos batch para identificar LS y ejecutarla cuando corresponda
            if args.metodo == "MOEAD+mKNNG-LS" and num_evaluaciones < args.num_evals_max:  
                # ¿cómo hace la selección de hijos para LS para un batch de 8 hijos y fracc=0.25?. 
                '''
                Cada subproblema i tiene un vector de pesos W[i] = [λerror, λgenes]
                g (Tchebycheff) mide qué tan lejos del ideal está un hijo, según los pesos de SU subproblema. 
                   g bajo = hijo muy bueno para su subproblema. 
                   g alto = malo.
                pos (k_esimo)	peso W=[λerr, λgen]	(error, genes)	g
                0	            [0.18, 0.82]	    (0.45, 35)	    0.070
                1	            [0.45, 0.55]	    (0.12, 60)	    0.165
                2	            [0.82, 0.18]	    (0.11, 110)	    0.144
                3	            [0.09, 0.91]	    (0.50, 33)	    0.040 ← mejor
                4	            [0.64, 0.36]	    (0.40, 50)	    0.213 ← peor
                5	            [0.27, 0.73]	    (0.15, 40)	    0.073
                6	            [0.91, 0.09]	    (0.105, 120)	0.081
                7	            [0.55, 0.45]	    (0.13, 45)	    0.068                
                Ejemplo de cálculo:
                Supongamos que:
                z (ideal o hasta ahora) = [error=0.10, genes=30]  # mínimos logrados hasta ahora (valores independientes) en todas las generaciones
                z_nad = [1.0, 130] → denominadores: 1-error= 0.90, 130-genes= 100
                Luego g se calcula:
                pos 0: max(0.18·(0.45-0.10)/0.90, 0.82·(35-30)/100) = max(0.070, 0.041) = 0.070
                pos 4: max(0.64·(0.40-0.10)/0.90, 0.36·(50-30)/100) = max(0.213, 0.072) = 0.213
                Por lo tanto al calcular k_ls_batch = round(0.25·8) = 2 => se eligen 2 con mejor g => pos 3 y 7 
                '''

                g_hijos = tchebycheff_vecinos(fitness_hijos, meta_hijos[:, 1], W, z, z_nad)
                hijos, fitness_hijos, evals_LS, diags_ls  = lsops.mkNNG_LS(args, hijos, num_evaluaciones, fitness_hijos, 
                                                                        seed_ls, generacion, ls_indptr, ls_indices, ls_P, X_no_scaled, 
                                                                        targetY, splits, seed_clf, inds_comunidades, comunidad_of_nodos, 
                                                                        top_nodes_por_comunidad, state["count_no_improve_hv0"], ctx_key,
                                                                        meta_hijos, g_hijos) ## , intervalo_LS ya no va,meta_hijos contiene: k_esimo_subprob_recorrido, num_subproblema
                diags.extend(diags_ls)
                num_evaluaciones += int(evals_LS.sum())
            #end-if

            # EP: guardar los hijos evaluados (post-LS) de este batch
            hijos_gen.append(hijos.copy())
            fit_gen.append(np.asarray(fitness_hijos, dtype=float))

            # re-calcular z y actualizar vecinos si corresponde
            for idx_hijo, (k_esimo_subprob_recorrido, num_subproblema) in enumerate(meta_hijos):
                hijo = hijos[idx_hijo][None, :]
                fitness_hijo = np.asarray(fitness_hijos[idx_hijo], dtype=float)
                # --- Step 2.3: actualizar z
                z = np.minimum(z, fitness_hijo)
                # --- Step 2.4: actualizar los vecinos B(i). Se usa max en Tchebycheff porque MOEA/D-Tchebycheff minimiza el peor desvío ponderado respecto al ideal ---
                vecinos = B[num_subproblema]
                # tchebycheff entrega los fitnees multiobjetivo usando f1, f2 con sus respectivos pesos por solución (o subproblema)
                # esta estrategia entrega el mayor de ellos al mutiplicarlo por los pesos (no hace la suma ponderada)
                #  recomendación paper (calcular el peor desbalance ponderado respecto al punto ideal.)
                # La idea es: una solución es buena para ese subproblema sólo si controla bien ambos objetivos según sus pesos.
                #             evitando que una solución “se salve” sólo porque es muy buena en un objetivo pero muy mala en el otro.
                g_hijo_vec = tchebycheff_vecinos(fitness_hijo,vecinos,W, z, z_nad)
                g_actual_vec = tchebycheff_vecinos(fitness_pop[vecinos],vecinos,W, z, z_nad) 
                mejora = g_hijo_vec < g_actual_vec #identifica qué vecinos mejora el hijo: gana el mejor (el de menor valor) de los máximos retornados por tchebycheff
                vecinos_reemplazo = vecinos[mejora] #[:args.moead_NR]
                if vecinos_reemplazo.size > 0:
                    population[vecinos_reemplazo] = hijo
                    fitness_pop[vecinos_reemplazo] = fitness_hijo
            #end-for
        #end-FOR

        # EP (Step 2.5): fusionar lo evaluado esta generación al archivo no-dominado
        if hijos_gen:
            arch_pop, arch_fit = actualizar_archivo(arch_pop, arch_fit,
                                                    np.vstack(hijos_gen), np.vstack(fit_gen))
        generacion += 1
        fronts = nds.do(fitness_pop)
        _, hv0 = control_metrics(fitness_pop, fronts, hv_tool)
        state = update_no_improve(state, hv0)
        dup_child = 1 - len(set(row_ids_bool(population))) / population.shape[0]  #no tenemos  una población de hijos

        if args.isOptuna == 'no':
            log_full(generacion, num_evaluaciones, fitness_pop, hv0, state, dup_child, diags)

    # END WHILE

    # =========================== SALIDA DEL FRENTE ===========================
    # SALIDA (A) — ACTIVA: frente de la POBLACIÓN final (MOEA/D estándar, sin EP).
    """
    packed = np.packbits(population.astype(np.bool_), axis=1)
    _, uniq_idx = np.unique(packed, axis=0, return_index=True)
    sol_frente = population[uniq_idx]
    fit_frente = fitness_pop[uniq_idx]
    idxs = nds.do(fit_frente)[0]                              # frente 0 (no-dominados)
    sol_frente, fit_frente = sol_frente[idxs], fit_frente[idxs]
    orden = np.lexsort((fit_frente[:, 0], fit_frente[:, 1]))  # ordenar por #genes asc.
    sol_frente, fit_frente = sol_frente[orden], fit_frente[orden]
    F0 = np.asarray(fit_frente, dtype=float).copy()
    F0[:, 0] = np.clip(F0[:, 0], 0.0, 1.0)
    F0_hv = float(hv_tool.do(F0))
    """
    # SALIDA (B) — DESACTIVADA: frente del ARCHIVO EXTERNO (EP).
    # Para usar el EP: descomenta este bloque + los 4 puntos "# [EP]" de arriba,
    # y comenta la SALIDA (A).
    arch_pop, arch_fit = actualizar_archivo(arch_pop, arch_fit, population, fitness_pop)
    orden = np.lexsort((arch_fit[:, 0], arch_fit[:, 1]))
    sol_frente, fit_frente = arch_pop[orden], arch_fit[orden]
    F0 = np.asarray(fit_frente, dtype=float).copy()
    F0[:, 0] = np.clip(F0[:, 0], 0.0, 1.0)
    F0_hv = float(hv_tool.do(F0))
    if args.isOptuna == 'no':
        print(f"[FRENTE pob.] {sol_frente.shape[0]} soluciones no-dominadas, "
              f"#genes en [{int(fit_frente[:,1].min())}, {int(fit_frente[:,1].max())}], "
              f"mejor f1={float(fit_frente[:,0].min()):.6f}")

    if trial is not None and last_reported != num_evaluaciones:
        running_mean_final = (args.completed_hv_sum + F0_hv) / (args.completed_count + 1)
        trial.report(running_mean_final, args.step_offset + num_evaluaciones)

    return sol_frente, fit_frente, F0_hv
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import EVALfitness.eval as eval 
import time
from joblib import Parallel, delayed 


def participation_coefficient(
    indptr: np.ndarray, indices: np.ndarray, communities: np.ndarray
) -> np.ndarray:
    """Calcula el coeficiente de participación de cada nodo desde una matriz CSR."""
    node_count = len(indptr) - 1
    if communities.size != node_count:
        raise ValueError("La cantidad de comunidades no coincide con los nodos del grafo.")

    coefficients = np.zeros(node_count, dtype=float)
    for node in range(node_count):
        neighbors = indices[indptr[node] : indptr[node + 1]]
        if neighbors.size == 0:
            continue
        _, counts = np.unique(communities[neighbors], return_counts=True)
        proportions = counts / neighbors.size
        coefficients[node] = 1.0 - np.square(proportions).sum()
    return coefficients


# --- Criterios de aceptación --------
def accept_drop(err_new, genes_new, err_old, genes_old, eps):
    if err_new <= err_old + eps and genes_new < genes_old:
        return True
    if err_new < err_old - eps and genes_new <= genes_old:
        return True
    return False

def accept_swap(err_new, genes_new, err_old, genes_old, eps):
    return err_new < err_old - eps and genes_new == genes_old

"""
FUNCIÓN: get_community_positions
    com_act: comunidad de cada activo
    pos_act: posición en inds_sorted (solo válidas)
    idx_act_valid: indices_activos alineados a pos_act (solo válidos)
"""
#com de cada activo, su posición en inds_sorted, y los activos válidos alineados
def get_community_positions(comunidad_of_nodos, indices_activos, inds_sorted):
    com_act = comunidad_of_nodos[indices_activos]
    pos = np.searchsorted(inds_sorted, com_act)
    valid = (pos >= 0) & (pos < inds_sorted.size) & (inds_sorted[pos] == com_act)
    return com_act[valid], pos[valid], indices_activos[valid]

def vecinos_activos_intercomunidad(indptr, indices, comunidad_of_nodos, idx_com, indices_activos):
    """Vecinos activos del primer nodo activo de idx_com que estén en OTRA comunidad.
    Usa adyacencia CSR (indptr/indices)"""
    if indptr is None:
        return np.array([], dtype=np.int32)
    genes_c = indices_activos[comunidad_of_nodos[indices_activos] == idx_com]
    if genes_c.size == 0:
        return np.array([], dtype=np.int32)
    v = int(genes_c[0])
    nbrs = indices[indptr[v]:indptr[v + 1]] # vecinos de v en O(grado)
    if nbrs.size == 0:
        return np.array([], dtype=np.int32)
    activos = set(indices_activos.tolist())
    out = [u for u in nbrs.tolist()
           if u in activos and comunidad_of_nodos[u] != idx_com]
    return np.asarray(out, dtype=np.int32)


def SWAP_inter(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, clasificador,
               indptr, indices, comunidad_of_nodos, P_g, counts_act, inds_sorted,
               ctx_key, max_evals, noImpHV, LS_frac_children ):  
    """SWAP de ESCAPE inter-comunidad. """
    num_eval = 0
    n_prob = 0
    activos = np.flatnonzero(sol)
    if activos.size == 0 or max_evals <= 0:
        return None, None, 0, 0, 0
    active_mask = np.zeros(sol.size, dtype=bool)
    active_mask[activos] = True
    # --- SALE: activos ordenados por menos vecinos ACTIVOS (aislados del activo, primero) ---
    deg_act = np.empty(activos.size, dtype=np.int32)
    for k, g in enumerate(activos):
        nb = indices[indptr[g]:indptr[g + 1]]
        deg_act[k] = int(active_mask[nb].sum()) if nb.size else 0
    orden_out = activos[np.argsort(deg_act, kind="stable")] # 0 vecinos activos = aislado → primero
    # --- ENTRA: inactivos de comunidades sin activos, ordenados por P descendente ---
    inactivos = np.flatnonzero(~sol)
    com_in = comunidad_of_nodos[inactivos]
    pos_in = np.searchsorted(inds_sorted, com_in)
    pos_ok = (pos_in < inds_sorted.size) & (inds_sorted[pos_in] == com_in)
    inactivos, pos_in = inactivos[pos_ok], pos_in[pos_ok]
    cand_in = inactivos[counts_act[pos_in] == 0]   # comunidades NO representadas
    if cand_in.size == 0:
        cand_in = inactivos    
    if cand_in.size == 0:
        return None, None, 0, 0, 0
    orden_in = cand_in[np.argsort(-P_g[cand_in], kind="stable")] # transversales (P alto) primero

    # ---INTENTOS: escala con noImpHV  ---
    top_out = max(1, int(round(LS_frac_children  * orden_out.size))) 
    top_in = top_out
    factor = 1 + noImpHV // 50     # cada 50 generaciones sin mejora
    orden_out = orden_out[:top_out * factor]
    orden_in  = orden_in[:top_in * factor]
    n_pares_disp = orden_out.size * orden_in.size
    max_intentos = min(n_pares_disp, max_evals)
    #print(f"n_pares_disp={n_pares_disp}, max_intentos={max_intentos}")

    # --- probar pares en orden de rank ---
    MAX_FALLOS = 10
    fallos = 0
    for g_out in orden_out:
        for g_in in orden_in:
            if n_prob >= max_intentos or num_eval >= max_evals or fallos >= MAX_FALLOS:
                return None, None, num_eval, n_prob, n_pares_disp
            new_sol = sol.copy()
            new_sol[int(g_out)] = False
            new_sol[int(g_in)] = True
            n_prob += 1
            (err_new, ngen_new), ev = eval.fitness_cached(
                new_sol, datax, targety, splits, clf_semilla, clasificador, ctx_key)
            num_eval += ev
            if accept_swap(err_new, ngen_new, F_sol[0], F_sol[1], epsilon):
            #if err_new < F_sol[0] - epsilon:         # estrictamente mejor (margen ε)
                return new_sol, np.array([err_new, ngen_new]), num_eval, n_prob, n_pares_disp
            fallos += 1 
    return None, None, num_eval, n_prob, n_pares_disp


def DROP(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, clasificador, ctx_key, max_evals, 
         idx_act, com_act, counts_act, inds_sorted, sobre_mitad_ppto, empty, indptr, indices, comunidad_of_nodos, indices_activos):
    import time
    t0 = time.perf_counter()
    num_eval = 0
    n_probados = 0  # 
    def obt_genes_act_star(idx_com):
        return idx_act[com_act == idx_com]
    # Por default: comunidad con más activos (inicialización)
    # (pero puede darse que la comunidad con más activos tenga solo uno)
    pos_com_act_star = int(np.argmax(counts_act))
    idx_com_act_star = int(inds_sorted[pos_com_act_star])
    genes_act_star = empty
    if sobre_mitad_ppto: # NACE LA SELECCIÓN ALEATORIA de de pos_com_act_star (comunidad). No selecciona la comunidad más grande
        pos_coms_validas = np.flatnonzero(counts_act >= 1)
        if pos_coms_validas.size > 0:
            pos_com_act_star = int(rng.choice(pos_coms_validas))
            idx_com_act_star = int(inds_sorted[pos_com_act_star])
            if (counts_act[pos_com_act_star] < 2):    # si hemos gastado más de la mitad del ppto AND
                genes_act_star = vecinos_activos_intercomunidad(indptr, indices, comunidad_of_nodos, idx_com_act_star, indices_activos)
                #
            else: #más de dos activos que  2 
                genes_act_star = obt_genes_act_star(idx_com_act_star)
        # 
    else: # presupuesto OK  (mientras ok con prespuesto nO se activa el aleatorio
        if (counts_act[pos_com_act_star] > 1 ):  
            #se usa: pos_com_act_star y idx_com_act_star  por defecto (con la máxima cantidad de activos )
            genes_act_star = obt_genes_act_star(idx_com_act_star) 

    genes_act_star = np.asarray(genes_act_star)
    if genes_act_star.size == 0:
        genes_drop = np.flatnonzero(sol).astype(np.int32, copy=False)
    else:
        genes_drop = genes_act_star.copy()
    
    rng.shuffle(genes_drop)
    drop_disp = len(genes_drop)
    # probar genes de la comunidad seleccionada
    for g_out in genes_drop:
        if num_eval >= max_evals:
            break        
        new_sol = sol.copy()
        new_sol[int(g_out)] = False
        if new_sol.sum() == 0:
            continue
        n_probados += 1
        (err_new, ngen_new), ev = eval.fitness_cached(new_sol, datax, targety, splits, clf_semilla, clasificador, ctx_key)
        num_eval += ev
        if accept_drop(err_new, ngen_new, F_sol[0], F_sol[1], epsilon):
            return new_sol, np.array([err_new, ngen_new]), num_eval, n_probados, drop_disp

    return None, None, num_eval, n_probados, drop_disp


def SWAP(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, clasificador, ctx_key, max_evals, 
         comunidad_of_nodos, inds_sorted, counts_act, top_nodes_por_comunidad, idx_act,com_act):  
    import time
    t0 = time.perf_counter()
    num_eval = 0
    n_pares_probados = 0
    pares_totales = 0
    indices_notactivos = np.flatnonzero(~sol)
    com_notact, pos_notact, idx_notact = get_community_positions(comunidad_of_nodos, indices_notactivos, inds_sorted  )
    if pos_notact.size > 0:
        counts_notact = np.bincount(pos_notact, minlength=inds_sorted.size)
        pos_out_validas = np.flatnonzero(counts_act >= 2)
        if pos_out_validas.size > 0:
            # preferir comunidades de entrada que hoy no tengan activos
            pos_in_preferidas = np.flatnonzero((counts_act == 0) & (counts_notact > 0))
            if pos_in_preferidas.size > 0:
                pos_in_pool = pos_in_preferidas
            else:
                pos_in_pool = np.flatnonzero(counts_notact > 0)
            if pos_in_pool.size > 0:
                pos_out = int(rng.choice(pos_out_validas))
                pos_in_candidatas = pos_in_pool[pos_in_pool != pos_out]
                if pos_in_candidatas.size > 0:
                    pos_in = int(rng.choice(pos_in_candidatas))
                    idx_com_out = int(inds_sorted[pos_out])
                    idx_com_in  = int(inds_sorted[pos_in])
                    # top-K nodos centrales de la comunidad de entrada (para guiar SWAP-in).
                    top_k_in = np.asarray(top_nodes_por_comunidad.get(idx_com_in, np.array([], dtype=np.int32)), dtype=np.int32)
                    genes_out = idx_act[com_act == idx_com_out].copy()
                    genes_in = idx_notact[com_notact == idx_com_in].copy()
                    n_pares_probados = 0  # ← pares candidatos realmente probados
                    if len(genes_out) == 0 or len(genes_in) == 0 or max_evals <= 0:
                        return None, None, 0, 0,0
                    if top_k_in.size > 0:
                        top_set = set(int(x) for x in top_k_in.tolist())
                        genes_in_prior = np.array([int(g) for g in genes_in if int(g) in top_set], dtype=np.int32)
                        if genes_in_prior.size == 0:
                            genes_in_prior = np.asarray(genes_in, dtype=np.int32).copy()
                    else:
                        genes_in_prior = np.asarray(genes_in, dtype=np.int32).copy()
                    # Aleatorizar el orden para que distintas LS exploren pares distintos
                    # (mantiene diversidad sin alterar el sesgo top-K ya aplicado).
                    genes_out_s = np.asarray(genes_out, dtype=np.int32).copy()
                    rng.shuffle(genes_out_s); rng.shuffle(genes_in_prior)
                    pares_totales  = len(genes_out_s) * len(genes_in_prior)
                    for g_out in genes_out_s:
                        for g_in in genes_in_prior:
                            if num_eval >= max_evals: 
                                return None, None, num_eval, n_pares_probados, pares_totales            
                            new_sol = sol.copy()
                            new_sol[int(g_out)] = False
                            new_sol[int(g_in)] = True
                            n_pares_probados += 1
                            (err_new, ngen_new), ev_fit = eval.fitness_cached(new_sol, datax, targety, splits, clf_semilla, clasificador, ctx_key )
                            num_eval += ev_fit
                            if accept_swap(err_new, ngen_new, F_sol[0], F_sol[1], epsilon):
                                return new_sol, np.array([err_new, ngen_new]), num_eval, n_pares_probados, pares_totales
    return None, None, num_eval, n_pares_probados, pares_totales


# ==================================================================

def LS_comunidades2(indptr, indices, P_g, rng, sol, F_sol, datax, targety, epsilon, splits, clf_semilla, clasificador, 
                    inds_comunidades,comunidad_of_nodos, top_nodes_por_comunidad, ctx_key,
                    sobre_mitad_ppto, max_evals, noImpHV,LS_frac_children):
    diag = {"msg_salida": "", "ev_drop": 0, "ev_swap": 0, "ev_inter": 0,
            "drop_disp": 0, "swap_pares_disp": 0, "inter_pares_disp": 0,
            "drop_ok": False, "swap_ok": False, "inter_ok": False}
    num_eval_g = 0
    empty = np.array([], dtype=np.int32)

    def ret(s, F):
        ls_ok = diag["drop_ok"] or diag["swap_ok"] or diag["inter_ok"]
        return s, F, num_eval_g, ls_ok, diag["swap_ok"], diag
    
    if sol.sum() == 0 or max_evals <= 0:
        diag["msg_salida"] = "sol_vacia_o_sin_ppto" 
        return ret(sol, F_sol.copy())
    indices_activos = np.flatnonzero(sol)
    inds_sorted = np.unique(np.asarray(inds_comunidades, dtype=np.int32)) # IDs de comunidades ordenados (necesario para searchsorted vectorizado).
    if inds_sorted.size == 0:
        diag["msg_salida"] = "sin_comunidades"; return ret(sol, F_sol.copy())
     
    com_act, pos_act, idx_act = get_community_positions(comunidad_of_nodos, indices_activos, inds_sorted) # paso índices activos
    if pos_act.size == 0:
        diag["msg_salida"] = "activos_sin_comunidad"; return ret(sol, F_sol.copy())
    counts_act = np.bincount(pos_act, minlength=inds_sorted.size)

    #.---------------------------------------------------------------------
    # DROP
    #.------------------------------------------------------------------------------------
    new_sol, new_F, ev, _, drop_disp = DROP(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, clasificador, 
                                          ctx_key, max_evals, idx_act, com_act, counts_act, inds_sorted, sobre_mitad_ppto, 
                                          empty, indptr, indices, comunidad_of_nodos, indices_activos)
    num_eval_g += ev
    diag["ev_drop"] = ev
    diag["drop_disp"] = int(drop_disp)
    if new_sol is not None:
        diag["drop_ok"] = True; diag["msg_salida"] = "drop_mejoro"
        return ret(new_sol, new_F)

    #.------------------------------------------------------------------------------------
    # SWAP: 
    #.------------------------------------------------------------------------------------
    rem = max_evals - num_eval_g
    if rem > 0:
        new_sol, new_F, ev, _, n_pares_disp = SWAP(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, 
                                                         clasificador, ctx_key, rem, comunidad_of_nodos, inds_sorted, 
                                                         counts_act, top_nodes_por_comunidad, idx_act,com_act)    
        num_eval_g += ev
        diag["ev_swap"] = ev; diag["swap_pares_disp"] = n_pares_disp
        if new_sol is not None:
            diag["swap_ok"] = True; diag["msg_salida"] = "swap_mejoro"; 
            return ret(new_sol, new_F)
        
    #.------------------------------------------------------------------------------------
    #  SWAP inter comunidades (ESCAPE: si no encuentra en las anteriores)
    # --------------------------------
    rem = max_evals - num_eval_g
    if rem > 0:
        new_sol, new_F, ev, _, n_pares_d = SWAP_inter(rng, sol, F_sol, epsilon, datax, targety, splits, clf_semilla, 
                                                  clasificador, indptr, indices, comunidad_of_nodos, P_g, 
                                                  counts_act, inds_sorted, ctx_key, rem, noImpHV,LS_frac_children)  
        num_eval_g += ev
        diag["ev_inter"] = ev
        diag["inter_pares_disp"] = n_pares_d
        if new_sol is not None:
            diag["inter_ok"] = True; diag["msg_salida"] = "inter_mejoro"
            return ret(new_sol, new_F)
    
    diag["msg_salida"] = "FIN: sin_mejora/sin_ppto"
    return ret(sol, F_sol.copy())


# ==================================================
#
#       PROPUESTA: mkNNG_LS
# 
# =======================================================
def mkNNG_LS(args, children, num_evaluaciones, fitness_children, seed_ls, generacion, ls_indptr, ls_indices,
             ls_P, X_no_scaled, targetY, splits, seed_clf, inds_comunidades, comunidad_of_nodos, top_nodes_por_comunidad, 
             count_no_improve_hv0, ctx_key, meta_hijos, g_hijos): # , intervalo_LS ya no va,meta_hijos contiene: k_esimo_subprob_recorrido, num_subproblema
    
    sobre_mitad_ppto = num_evaluaciones > 0.5 * args.num_evals_max # para saber si pasamos el 70% de las evaluaciones
    remaining = max(0, args.num_evals_max - num_evaluaciones)

    if args.metodo == "NSGA-II+mKNNG-LS":
        k_ls = max(1, int(round(args.LS_frac_children * children.shape[0])))           
        elite_idx = np.argsort(fitness_children[:, 0])[:k_ls]         
        mask_ls = np.zeros(children.shape[0], dtype=bool)
        mask_ls[elite_idx] = True
        divisor_presupuesto = k_ls
        child_ss = np.random.SeedSequence(seed_ls + generacion).spawn(k_ls) #np.random.SeedSequence(seed_ls + generacion).spawn(pop_ls.shape[0])

    if args.metodo == "MOEAD+mKNNG-LS": 
        meta_hijos = np.asarray(meta_hijos, dtype=int)        
        k_ls_batch  = max(1, int(round(args.LS_frac_children * children.shape[0])))   # ADENTRO
        idx_mejores = np.argsort(g_hijos)[:k_ls_batch]
        mask_ls = np.zeros(children.shape[0], dtype=bool)
        mask_ls[idx_mejores] = True
        k_ls = int(mask_ls.sum())
        if k_ls == 0:
            return children, fitness_children, np.array([], dtype=int), []
        idx_ls = np.where(mask_ls)[0]
        primer_k_esimo = int(meta_hijos[idx_ls[0], 0]) 
        subprob_restantes = max(k_ls, int(round(args.LS_frac_children * args.pop_size)))

        divisor_presupuesto = subprob_restantes
        child_ss = np.random.SeedSequence([seed_ls, generacion, primer_k_esimo]).spawn(k_ls)

    if remaining < k_ls:
        return children, fitness_children, np.array([], dtype=int), []
    budget_per_ls = remaining // divisor_presupuesto  ## Reparto del presupuesto por adelantado (las LS corren en paralelo y no pueden compartir un contador vivo).
    if budget_per_ls <= 0:
        return children, fitness_children, np.array([], dtype=int), []
    
    pop_ls = children[mask_ls]
    fit_ls = fitness_children[mask_ls]
    results_LS = Parallel(n_jobs=max(1, args.n_jobs_ls), backend="threading", verbose=0)(
        delayed(LS_comunidades2)(ls_indptr, ls_indices, ls_P, np.random.default_rng(child_ss[i]), sol, fit_ls[i], X_no_scaled, targetY,
                                        args.epsilon, splits, seed_clf, args.clasificador, inds_comunidades, comunidad_of_nodos, 
                                        top_nodes_por_comunidad, ctx_key, sobre_mitad_ppto, budget_per_ls, 
                                        count_no_improve_hv0, args.LS_frac_children)
        for i, sol in enumerate(pop_ls)
    )     
    pop_ls_new, fit_ls_new, evals_LS, ls_ok_arr, swap_ok_arr, diags = zip(*results_LS)
    pop_ls_new  = np.stack(pop_ls_new)
    fit_ls_new  = np.stack(fit_ls_new)
    evals_LS    = np.array(evals_LS,  dtype=int)
    diags       = list(diags)
    children[mask_ls] = pop_ls_new
    fitness_children[mask_ls] = fit_ls_new

    return children, fitness_children, evals_LS, diags





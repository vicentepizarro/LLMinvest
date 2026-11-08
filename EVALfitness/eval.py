# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed 
from typing import Dict, Tuple
import hashlib
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threading import RLock

# Cachés POR-PROCESO (bajo loky cada worker tiene los suyos; las claves incluyen
# ctx_key, así que nunca hay colisión entre datasets).
fitness_cache: Dict[Tuple[bytes, bytes], Tuple[float, int]] = {}
fold_stats_cache: Dict[bytes, list] = {}      # ctx_key -> [(mu_k, sd_k), ...]
cache_lock = RLock()  #Lock()


def sol_key(sol):
    s = np.ascontiguousarray(sol.astype(np.bool_, copy=False), dtype=np.bool_)
    packed = np.packbits(s, axis=None).tobytes()
    return hashlib.blake2b(packed, digest_size=16).digest()

"""
Construye un identificador estable del contexto de evaluación.
Dos llamadas con el mismo contexto comparten caché; con cualquier
diferencia (dataset, clasificador, splits) el caché las trata como independientes.
"""
def context_key(dataset_id, clasificador, clf_semilla, cv_folds):
    payload = (f"ds={dataset_id}|clf={clasificador}|clf_seed={int(clf_semilla)}|"
        f"folds={int(cv_folds)}").encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()

def crear_clasificador(clasificador, clf_semilla):
    if clasificador == "NB": 
        return GaussianNB()
    if clasificador == "RF":
        return RandomForestClassifier(random_state=clf_semilla, n_jobs=1, class_weight="balanced")
    if clasificador == "DT":
        return DecisionTreeClassifier(random_state=clf_semilla, class_weight="balanced", max_depth=5, min_samples_leaf=5)
    if clasificador == "ET":
        return ExtraTreeClassifier(random_state=clf_semilla, class_weight="balanced")
    if clasificador == "EXT": 
        return ExtraTreesClassifier(n_estimators=50, max_features="sqrt", bootstrap=False, random_state=clf_semilla, n_jobs=1, class_weight="balanced")
    if clasificador == "RL": #1
        return LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=clf_semilla, class_weight="balanced") #max_iter=1000,
    if clasificador == "LDA": #2
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if clasificador == "SGD": #3
        return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=clf_semilla, class_weight="balanced")
    raise ValueError(f"Clasificador no reconocido: {clasificador}")

def preparar_stats_por_fold(xdata_NO_scaled, splits):
    fold_stats = []
    for tr, te in splits:
        mu = xdata_NO_scaled[tr].mean(axis=0).astype(np.float32, copy=False)
        sd = xdata_NO_scaled[tr].std(axis=0).astype(np.float32, copy=False)
        sd[sd == 0.0] = 1.0
        fold_stats.append((mu, sd))
    return fold_stats

def get_fold_stats(xdata_NO_scaled, splits, ctx_key):
    """Estadísticas de fold cacheadas POR CONTEXTO. Se calculan una sola vez por
    dataset en cada proceso (también en los workers loky): elimina el recálculo
    por generación y la dependencia de globals."""
    with cache_lock:
        st = fold_stats_cache.get(ctx_key)
    if st is not None:
        return st
    st = preparar_stats_por_fold(xdata_NO_scaled, splits)
    with cache_lock:
        fold_stats_cache.setdefault(ctx_key, st)
        return fold_stats_cache[ctx_key]

'''
FUNCIÓN: fitness_calidad_numFeatures
Escala por fold usando fold_mu/fold_sd precalculados (SIN leakage).
Se usa mu/sd del TRAIN para escalar TRAIN y TEST en cada fold
'''
def fitness_calidad_numFeatures_con_stats(sol_bin, xdata_NO_scaled, ytarget, splits, clf_semilla, clasificador, fold_stats):
    """Escala TRAIN y TEST por fold con mu/sd de TRAIN (sin leakage). f1=1-AUC, f2=#genes."""
    cols = np.flatnonzero(sol_bin)
    if cols.size == 0:
        return 1.0, 0
    X_subset = xdata_NO_scaled[:, cols]
    clf = crear_clasificador(clasificador, clf_semilla)
    aucs = []
    for k, (tr, te) in enumerate(splits):
        mu = fold_stats[k][0][sol_bin]
        sd = fold_stats[k][1][sol_bin]
        Xtr = (X_subset[tr] - mu) / sd
        Xte = (X_subset[te] - mu) / sd
        clf.fit(Xtr, ytarget[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(ytarget[te], proba))
    return 1.0 - float(np.mean(aucs)), int(sol_bin.sum())

def fitness_cached(sol_bin, xdata, ytarget, splits, clf_semilla, clasificador, ctx_key):
    """Evalúa con caché por-proceso. Usado por la LS dentro de los workers loky."""
    full_key = (sol_key(sol_bin), ctx_key)
    with cache_lock:
        hit = fitness_cache.get(full_key)
    if hit is not None:
        return hit, 0
    fold_stats = get_fold_stats(xdata, splits, ctx_key)
    valor = fitness_calidad_numFeatures_con_stats(sol_bin, xdata, ytarget, splits,
                                                  clf_semilla, clasificador, fold_stats)
    with cache_lock:
        fitness_cache[full_key] = valor
    return valor, 1

def eval_batch_with_cache_loky(solutions, xdata, ytarget, splits, clf_semilla, clasificador, n_jobs, ctx_key):
    """Chequea caché central (proceso principal), evalúa en paralelo SOLO lo
    faltante (loky), actualiza caché y devuelve (fitness_array, num_evals_reales)."""
    sols = [np.ascontiguousarray(sol.astype(np.bool_, copy=False)) for sol in solutions]
    full_keys = [(sol_key(s), ctx_key) for s in sols]
    fitness = [None] * len(sols)
    to_eval_idx, to_eval_sols  = [], []
    with cache_lock:
        for i, fk in enumerate(full_keys):
            val = fitness_cache.get(fk)
            if val is not None:
                fitness[i] = val
            else:
                to_eval_idx.append(i)
                to_eval_sols.append(sols[i])
    if to_eval_idx:
        fold_stats = get_fold_stats(xdata, splits, ctx_key) # 1 sola vez por dataset
        #results = Parallel(n_jobs=max(1, n_jobs), backend="loky", verbose=0)(
        results = Parallel(n_jobs=max(1, n_jobs), backend="threading", verbose=0)(
            delayed(fitness_calidad_numFeatures_con_stats)(sol, xdata, ytarget, splits, clf_semilla, clasificador, fold_stats)
            for sol in to_eval_sols
        )
        # Actualizar caché central y reconstruir
        with cache_lock:
            for idx, val in zip(to_eval_idx, results):
                fitness_cache[full_keys[idx]] = val
                fitness[idx] = val
    return np.array(fitness, dtype=float), len(to_eval_idx)

def eval_children_dedup(pop_children, X_scaled, targetY, splits, clf_semilla, clasificador, n_jobs, ctx_key):
    """Deduplica genotipos por fila antes de evaluar (evita gastar 2 evals en gemelos)."""
    pop_children = np.ascontiguousarray(pop_children.astype(np.bool_, copy=False))
    _, uniq_idx, inv = np.unique(pop_children, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel() 
    fit_uniq, evs = eval_batch_with_cache_loky(pop_children[uniq_idx], X_scaled, targetY, splits, clf_semilla, 
                                               clasificador, n_jobs=n_jobs, ctx_key=ctx_key)
    return fit_uniq[inv], evs

def clear_ls_caches():
    """Libera caché de fitness y de estadísticas (proceso principal). Necesario
    entre datasets en Optuna para acotar RAM y garantizar presupuesto justo por trial."""
    with cache_lock:
        fitness_cache.clear()
        fold_stats_cache.clear()

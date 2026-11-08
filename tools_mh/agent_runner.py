from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from Filtros.FS_metodos_filtros import FS_metodo_filter
from tools_mh.input_output_data import leer_dataset_cumida_2clases


def ejecutar_mono(
    algoritmo: str,
    dataset_path: str,
    *,
    correlation_threshold: float = 0.6,
    max_evaluations: int = 100,
    positive_label: str = "tumoral",
    negative_label: str = "normal",
    seed: int = 42,
) -> dict[str, Any]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    minimum_evaluations = 11 if algoritmo == "GA" else 2
    if max_evaluations < minimum_evaluations:
        raise ValueError(
            f"max_evaluations debe ser al menos {minimum_evaluations} "
            f"para {algoritmo}."
        )

    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    features, data, target = leer_dataset_cumida_2clases(
        str(path), separator, positive_label, negative_label
    )
    class_counts = np.bincount(np.asarray(target, dtype=int), minlength=2)
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError(
            "El dataset debe tener al menos dos muestras de cada clase "
            f"('{positive_label}' y '{negative_label}')."
        )

    args = Namespace(
        semilla=seed,
        umbral_correlacion=correlation_threshold,
        umbral_varianza=0.0,
        significancia=0.05,
        n_jobs_pv_rf=1,
        max_iterations_rf=100,
        cv_rf=RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=5, random_state=seed
        ),
        cv_fitness=StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed
        ),
        max_iterations_fitness=1000,
        n_jobs_fitness=1,
        num_eval_max=max_evaluations,
        num_vecinos_LS=20,
        gene_initial_prob=0.02,
        pop_size=10,
        percentage_select=0.1,
        crossover_rate=0.1,
        mutation_rate=0.2,
        gene_mutation_rate=0.05,
        isLS="si",
        isOptuna="no",
        ini_temp=1000.0,
        min_temp=0.001,
        alpha_func_cool=0.99,
        repeats_temp=20,
        max_size_bloque=3,
        porc_mutacion=0.002,
    )

    selected_indices, filtered_data = FS_metodo_filter(
        features, data, target, args
    )
    selected_features = np.asarray(features)[selected_indices]
    if filtered_data.shape[1] == 0:
        raise ValueError("El filtro estadístico no seleccionó ninguna característica.")

    algorithm_modules = {
        "GA": ("FS_optimizador_AG.algoritmo_genetico", "AG"),
        "SA": ("FS_optimizador_SA.simulated_annealing", "simulated_annealing"),
        "VNS": (
            "FS_optimizador_VNS.variable_neighborhood_search",
            "variable_neighborhood_search",
        ),
    }
    module_name, function_name = algorithm_modules[algoritmo]
    optimizer = getattr(importlib.import_module(module_name), function_name)

    if algoritmo == "GA":
        solution, score = optimizer(
            selected_features, filtered_data, target, args=args
        )
    else:
        solution, score = optimizer(
            selected_features, filtered_data, target, args=args
        )

    chosen_features = selected_features[np.asarray(solution, dtype=bool)]
    return {
        "algorithm": algoritmo,
        "dataset": str(path),
        "selected_features": chosen_features.tolist(),
        "selected_count": int(chosen_features.size),
        "score": float(score),
        "max_evaluations": max_evaluations,
    }

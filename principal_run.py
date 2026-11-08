from __future__ import annotations

import argparse
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from MKNN.MKNN import grafo_guiador_LS
from MOoptimization.MOEAD import MOEAD
from MOoptimization.NSGAII import NSGA2


SUPPORTED_METHODS = {
    "NSGA-II",
    "NSGA-II+mKNNG-LS",
    "MOEAD",
    "MOEAD+mKNNG-LS",
}
SUPPORTED_CLASSIFIERS = {"NB", "RF", "DT", "ET", "EXT", "RL", "LDA", "SGD"}


def leer_dataFilter_y_matrizSim(
    file_filter: str | Path, args: Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Carga la matriz prefiltrada y la similitud requerida por la búsqueda local."""
    sims: dict[str, np.ndarray] = {}
    with np.load(file_filter, allow_pickle=False) as data:
        required_keys = {"X", "genes", "Y"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise ValueError(
                f"El NPZ no contiene las claves requeridas: {', '.join(sorted(missing_keys))}."
            )
        x_filtered = data["X"]
        genes_filtered = data["genes"].ravel()
        target = data["Y"].ravel()
        if args.metodo.endswith("+mKNNG-LS"):
            sim_key = f"S_{args.similitud}"
            if sim_key not in data.files:
                raise ValueError(
                    f"El NPZ no contiene {sim_key}, requerido por {args.metodo}."
                )
            sims[args.similitud] = data[sim_key]
    return x_filtered, genes_filtered, target, sims


def ejecutar(
    dataset_path: str,
    *,
    metodo: str = "NSGA-II",
    clasificador: str = "RL",
    max_evaluations: int = 100,
    pop_size: int = 10,
    gene_initial_prob: float = 0.02,
    crossover_rate: float = 0.1,
    mutation_rate: float = 0.2,
    gene_mutation_rate: float = 0.05,
    ls_frac_children: float = 0.2,
    cv_folds: int = 5,
    epsilon: float = 0.0,
    seed: int = 42,
    k_optimo: int = 5,
    similitud: str = "pearson",
    n_jobs_fitness: int = 1,
    n_jobs_ls: int = 1,
    moead_t: int = 3,
    moead_batch_size: int = 4,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Ejecuta NSGA-II o MOEA/D sobre un NPZ previamente filtrado."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset filtrado: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("principal_run requiere un dataset filtrado en formato .npz.")
    if metodo not in SUPPORTED_METHODS:
        raise ValueError(
            f"Método '{metodo}' no válido. Opciones: {', '.join(sorted(SUPPORTED_METHODS))}."
        )
    if clasificador not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            "Clasificador no válido. Opciones: "
            + ", ".join(sorted(SUPPORTED_CLASSIFIERS))
            + "."
        )
    if max_evaluations < 1 or pop_size < 2 or cv_folds < 2:
        raise ValueError(
            "max_evaluations debe ser positivo, pop_size al menos 2 y cv_folds al menos 2."
        )
    if not 0 < gene_initial_prob < 1:
        raise ValueError("gene_initial_prob debe estar entre 0 y 1.")
    if not 1 <= moead_t <= pop_size:
        raise ValueError("moead_t debe estar entre 1 y pop_size.")
    if not 1 <= moead_batch_size <= pop_size:
        raise ValueError("moead_batch_size debe estar entre 1 y pop_size.")

    args = Namespace(
        metodo=metodo,
        clasificador=clasificador,
        nom_dataset=path.stem.removesuffix("_filter"),
        num_evals_max=max_evaluations,
        pop_size=pop_size,
        gene_initial_prob=gene_initial_prob,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        gene_mutation_rate=gene_mutation_rate,
        LS_frac_children=ls_frac_children,
        cv_folds=cv_folds,
        epsilon=epsilon,
        semilla=seed,
        k_optimo=k_optimo,
        similitud=similitud,
        n_jobs_fitness=n_jobs_fitness,
        n_jobs_ls=n_jobs_ls,
        moead_T=moead_t,
        moead_batch_size=moead_batch_size,
        isOptuna="no",
        run_id=0,
    )

    data, genes, target, similarities = leer_dataFilter_y_matrizSim(path, args)
    if data.ndim != 2 or target.ndim != 1 or data.shape[0] != target.shape[0]:
        raise ValueError("Las dimensiones de X e Y en el NPZ no son compatibles.")
    if data.shape[1] != genes.size:
        raise ValueError("La cantidad de columnas de X no coincide con genes.")
    if np.unique(target).size != 2:
        raise ValueError("El dataset debe contener exactamente dos clases.")
    if metodo.endswith("+mKNNG-LS") and k_optimo < 1:
        raise ValueError("k_optimo debe ser positivo para los métodos con mKNNG-LS.")

    graph = communities = community_ids = top_nodes = None
    if metodo.endswith("+mKNNG-LS"):
        graph_path = (
            path.parent
            / "graphs"
            / f"{args.nom_dataset}_{similitud}_{k_optimo}-MNN_grafo.gpickle"
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph, communities, community_ids, top_nodes = grafo_guiador_LS(
            seed,
            similarities[similitud],
            args.nom_dataset,
            k_optimo,
            genes,
            str(graph_path),
        )

    master_seed = seed
    random_cv = master_seed + 1
    splits = list(
        RepeatedStratifiedKFold(
            n_splits=cv_folds, n_repeats=1, random_state=random_cv
        ).split(data, target)
    )
    rng_global = np.random.default_rng(master_seed + 2)
    rng_init = np.random.default_rng(master_seed + 3)
    seed_ls = master_seed + 4
    seed_clf = master_seed + 5

    optimizer = MOEAD if metodo.startswith("MOEAD") else NSGA2
    start = time.monotonic()
    solutions, fitness_front, hypervolume = optimizer(
        graph,
        data,
        target,
        communities,
        community_ids,
        top_nodes,
        trial=None,
        args=args,
        splits=splits,
        seed_clf=seed_clf,
        rng_global=rng_global,
        rng_init=rng_init,
        seed_ls=seed_ls,
    )

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = path.with_name(f"{args.nom_dataset}_{metodo}_{timestamp}_pareto.csv")
    else:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as stream:
        stream.write("idx;error;num_genes;selected_genes\n")
        for index, solution in enumerate(solutions):
            selected_names = genes[np.flatnonzero(solution)].astype(str)
            stream.write(
                f"{index};{fitness_front[index, 0]:.10f};"
                f"{int(fitness_front[index, 1])};{';'.join(selected_names)}\n"
            )

    return {
        "status": "success",
        "algorithm": metodo,
        "dataset": str(path),
        "pareto_front_path": str(output),
        "pareto_solutions": int(len(solutions)),
        "hypervolume": float(hypervolume),
        "elapsed_seconds": time.monotonic() - start,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--metodo", choices=sorted(SUPPORTED_METHODS), default="NSGA-II")
    parser.add_argument(
        "--clasificador", choices=sorted(SUPPORTED_CLASSIFIERS), default="RL"
    )
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--gene-initial-prob", type=float, default=0.02)
    parser.add_argument("--crossover-rate", type=float, default=0.1)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--gene-mutation-rate", type=float, default=0.05)
    parser.add_argument("--ls-frac-children", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-optimo", type=int, default=5)
    parser.add_argument("--similitud", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--n-jobs-fitness", type=int, default=1)
    parser.add_argument("--n-jobs-ls", type=int, default=1)
    parser.add_argument("--moead-t", type=int, default=3)
    parser.add_argument("--moead-batch-size", type=int, default=4)
    parser.add_argument("--output-path")
    return parser.parse_args()


if __name__ == "__main__":
    print(ejecutar(**vars(_parse_args())))

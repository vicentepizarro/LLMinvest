from __future__ import annotations

import argparse

from tools_mh.agent_runner import ejecutar_mono


def ejecutar(
    dataset_path: str,
    corr_thresh: float = 0.6,
    max_evaluations: int = 100,
    positive_label: str = "tumoral",
    negative_label: str = "normal",
    seed: int = 42,
) -> dict:
    """Ejecuta el algoritmo genético de selección de características."""
    return ejecutar_mono(
        "GA",
        dataset_path,
        correlation_threshold=corr_thresh,
        max_evaluations=max_evaluations,
        positive_label=positive_label,
        negative_label=negative_label,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--corr-thresh", type=float, default=0.6)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--positive-label", default="tumoral")
    parser.add_argument("--negative-label", default="normal")
    parser.add_argument("--seed", type=int, default=42)
    print(ejecutar(**vars(parser.parse_args())))

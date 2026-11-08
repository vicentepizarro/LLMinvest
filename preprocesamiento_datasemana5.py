import os
import numpy as np
import pandas as pd

def preparar_dataset_benchmark(
        csv_path: str, output_dir: str = "data_processed"
):
    """
    Carga el dataset de CuMiDA, separa las etiquetas y calcula la matriz de correlación génica  (M_corr) neceraia para simLS.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cargando dataset desde: {csv_path}...")

    df = pd.read_csv(csv_path)

    if "type" in df.columns:
        label_col = "type"
    elif "Class" in df.columns:
        label_col = "Class"
    else:
        label_col = str(df.columns[6])

    X = df.drop(columns=[label_col])

    print(
        f"Dataset cargado correctamente: {X.shape[0]} muestras, {X.shape[1]} "
        "genes."
    )

    expr_path = os.path.join(output_dir, "M_expr.csv")
    df.to_csv(expr_path, index=False)

    print("Calculando matriz de correlación de Pearson entre genes...")
    M_corr = X.corr(method="pearson").abs()

    corr_path = os.path.join(output_dir, "M_corr.npy")
    np.save(corr_path, M_corr.values)

    print(
        f"Archivos guardados en: '{output_dir}'\n -"
        f" {expr_path}\n - {corr_path}"
    )

    return expr_path, corr_path


if __name__ == "__main__":
    DATASET_PATH = "/home/vicente/Descargas/Colorectal_GSE44861.csv" # Ruta de tu CSV CuMiDA
    if os.path.exists(DATASET_PATH):
        preparar_dataset_benchmark(DATASET_PATH)
    else:
        print(f"Coloca la ruta a tu CSV en DATASET_PATH. Actualmente apunta a: {DATASET_PATH}")
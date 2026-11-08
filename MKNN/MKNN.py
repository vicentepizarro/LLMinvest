import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") # fuerza backend sin GUI
from collections import Counter
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
import scipy.sparse as sp
import networkx as nx
from community import community_louvain 
import os
import gzip
import pickle
from scipy.stats import spearmanr


# VISULIZACIONES PANDAS y NUMPY
pd.set_option('display.max_rows', None) # # Mostrar todas las filas
pd.set_option('display.max_columns', None) # Mostrar todas las columnas
np.set_printoptions(threshold=np.inf)
#pd.set_option('display.max_colwidth', None) ## Aumentar el ancho de columnas si es necesario
pd.set_option('display.width', None) ## Opcional: desactiva el ancho máximo total de la tabla
semilla = 200 
np.random.seed(semilla)

def get_matriz_similitud(X_scaled, similitud_name,  gamma):
    #.-
    if similitud_name == "pearson":
        sim_matriz = np.corrcoef(X_scaled, rowvar=False)
    elif similitud_name == "spearman":
        sim_matriz, _ = spearmanr(X_scaled, axis=0)
    sim_matriz = np.clip(sim_matriz, 0, None)    # # signed-hybrid: negativas -> 0
    sim_matriz = sim_matriz ** gamma   
    np.fill_diagonal(sim_matriz, 0)         # auto-correlación = 0
    print("Corr min:", np.min(sim_matriz), "Sim max:", np.max(sim_matriz))
    return sim_matriz.astype(np.float32, copy=False)

#_____________________________________________________
#                  GRAFO
#___________________________________________________

def eliminar_aislados(H, s_min=3):
    H.remove_nodes_from(list(nx.isolates(H)))  ## 1) nodos aislados
    small = [c for c in nx.connected_components(H) if len(c) < s_min] ## 2) componentes muy pequeños
    for comp in small:
        H.remove_nodes_from(comp)
    return H
def obtener_cliques(H, min_size):
    #print("número de nodos antes (obtener_cliques):", H.number_of_nodes())
    H = eliminar_aislados(H, s_min=min_size)
    grado_min = min_size - 1 #Poda previa: nodos que jamás podrán formar cliques ≥ min_size
    H.remove_nodes_from([n for n, d in H.degree if d < grado_min])
    #print("número de nodos después (obtener_cliques):", H.number_of_nodes())
    cliques = [c for c in nx.find_cliques(H) if len(c) >= min_size]
    return cliques
def evaluar_k_MNN_grafo(semilla, G, s_min=3):
    partition = community_louvain.best_partition(G, weight="weight", random_state=semilla)
    comm_sizes = np.array(list(Counter(partition.values()).values()))
    Q = community_louvain.modularity(partition, G, weight="weight")
    Porc_iso = nx.number_of_isolates(G)/G.number_of_nodes()
    num_comptes_small = [c for c in nx.connected_components(G) if len(c) <= s_min]
    return partition, {"Q": Q, "Porc_iso": Porc_iso*100, "num_comptes": nx.number_connected_components(G), 
                       "num_comptes_small": len(num_comptes_small), "total_nodos": G.number_of_nodes(),
                       "num_nodos_iso": nx.number_of_isolates(G), "num_comudes": len(comm_sizes)}
def Kvecinos_cernanos(corr, k):
    n = corr.shape[0]
    kth = min(k, n-1) # Pivote: es mejor que range(1, k+1) (ya que no es necesario todos ordenados). kth es el pivote del ordenamiento Introselect" (una combinación de Quicksort y Heapsort para garantizar un rendimiento de peor caso O(NlogN)
    knn_idx = np.argpartition(-corr, kth=kth, axis=1)[:, :k] # - Dim (N, k) y knn_idx = np.argpartition(-corr, kth=kth, axis=1) Dim (N,N)
                                                             # - np.argpartition por defecto busca los más pequeños. Se multiplica por -1 para obtener los más grandes
                                                             #   (los más altos se convierten en los más bajos), (puede recuperar índices con +np.inf pero se eliminan después).
                                                             # - Los elementos a la izquierda de kth serán menores o igual a él y los de su derecha serán mayores o igual a él.
                                                             # - np.argpartition axis=1: En cada fila i se encuentran los k índices de las corrs más "bajas" (altas en este caso) de ese i.
    rows = np.repeat(np.arange(n), k) # (n*k,). Cada elemento de array np.arange(n) lo repite k veces # [0 0 … 1 1 … n-1 n-1] Dim: (n*k,)
    cols = knn_idx.ravel()            # convertir matriz en una dimensión: (n, k) → (n*k,)
    data = np.ones_like(rows, dtype=bool) # Todos True, (n*k,)
    mask = corr[rows, cols] > -np.inf       # mantener sólo los pesos finitos
    rows, cols, data = rows[mask], cols[mask], data[mask]
    vecindad_mutua = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr() ## matriz booleana dispersa de vecinos dirigidos
    return vecindad_mutua
def grafo_knn_mutuo(features, corr, k): #umbral,  procedimiento
    n = corr.shape[0]
    if k <= 0 or k >= n:
        raise ValueError("k debe ser un entero en el rango 1 … N-1")
    vecindad_mutua = Kvecinos_cernanos(corr, k)
    matriz_mknn = vecindad_mutua.minimum(vecindad_mutua.T)          # mantiene sólo las aristas simétricas
    i, j = matriz_mknn.nonzero()        
    upper = i < j                      # evita duplicados
    i, j = i[upper], j[upper]
    pesos = corr[i, j]
    mask_w = pesos > 0.0
    i, j, pesos = i[mask_w], j[mask_w], pesos[mask_w]
    G = nx.Graph()
    G.add_nodes_from(range(features.shape[0]))  #features
    G.add_weighted_edges_from(zip(i,j,pesos))  #features[i], features[j],
    #print(f"Grafo generado (k={k}) con {G.number_of_edges()} aristas y {G.number_of_nodes()} nodos")  
    return G
def save_graph_pickle(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:                     # compresión automática
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
def load_graph_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
def cargar_o_crear_grafo_K_MNN_con_mejor_k(semilla, file_grafo_disco, features_FILTER, sim_mat, nom_dataset, k_optimo):
    if os.path.exists(file_grafo_disco):
        print(f"Leyendo grafo desde {file_grafo_disco}") # ya fue evaluado con la función evaluar_k_MNN_grafo y save_metricas_grafo_mejor_k
        G_mutuo = load_graph_pickle(file_grafo_disco)
        partition, metricas = evaluar_k_MNN_grafo(semilla, G_mutuo, s_min=3)        
    else:
        print(f"Grafo no encontrado.... Creando Grafo {k_optimo}-MNN del dataset {nom_dataset}")
        G_mutuo = grafo_knn_mutuo(features_FILTER, sim_mat, k_optimo)
        save_graph_pickle(G_mutuo, file_grafo_disco)  #, protocol=pickle.HIGHEST_PROTOCOL
        partition, metricas = evaluar_k_MNN_grafo(semilla, G_mutuo, s_min=3)        
        print(f"Grafo guardado en {file_grafo_disco}")
    return G_mutuo, partition, metricas 


def crear_figura_tabla(tabla_df, k_optimo):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    tabla_plt = ax.table(cellText=tabla_df.round(3).astype(str).values, 
                        colLabels=tabla_df.columns, loc='upper center', cellLoc='center' )
    tabla_plt.auto_set_font_size(False)
    tabla_plt.set_fontsize(12)
    ncols = len(tabla_df.columns)
    for c in range(ncols):                # solo la fila 0
        tabla_plt[(0, c)].set_text_props(fontweight="bold", fontsize=13)
    tabla_plt.scale(1.2, 1.6)
    return fig
def save_metricas_grafo(df, pdf_path, titulo, ylim_Q, ylim_iso, k_optimo):
    print(df.round(3))
    with PdfPages(pdf_path) as pdf:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        fig.suptitle(titulo, fontsize=12, fontweight='bold')
        # Curva de modularidad
        ax1.plot(df["k"], df["Q"], marker="o", color="tab:blue", label="Modularidad Q")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Modularidad", color="tab:blue")
        ax1.set_ylim(*ylim_Q)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        # Curva de % aislados en eje secundario
        ax2 = ax1.twinx()
        ax2.plot(df["k"], df["Porc_iso"] * 100, marker="s", color="tab:orange", label="% aislados")
        ax2.set_ylabel("% aislados", color="tab:orange")
        ax2.set_ylim(*ylim_iso)
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        # Leyenda combinada
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        #ax1.set_title(titulo)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        #.-
        fig_tabla = crear_figura_tabla(df, k_optimo)
        pdf.savefig(fig_tabla)  # fig_tabla.savefig(tabla_pdf, bbox_inches="tight")
        plt.close(fig_tabla) #plt.close()
    return
def elegir_k_optimo(df_sim, ISO_UMBRAL, DELTA_Q_MAX):
    df_sim = df_sim.sort_values("k").reset_index(drop=True)
    for i in range(1, len(df_sim)):
        iso_ok   = df_sim.loc[i, "Porc_iso"] <= ISO_UMBRAL
        delta_q  = abs(df_sim.loc[i, "Q"] - df_sim.loc[i-1, "Q"])
        plateau  = delta_q <= DELTA_Q_MAX
        if iso_ok and plateau:
            return int(df_sim.loc[i, "k"])
    # fallback: k con mínimo % aislados (si nunca alcanzó la meseta)
    k_min_iso = int(df_sim.loc[df_sim["Porc_iso"].idxmin(), "k"])
    return k_min_iso

def evaluar_y_seleccionar_mejor_k(k_list, matrices_similitud, features_FILTER, args):
    results = []     
    for sim_name, sim_mat in matrices_similitud:    # [('pearson',S1), ('spearman',S2), ...]
        for k in k_list:  # raw_w = powered_w ** (1 / gamma)
            G_mutuo = grafo_knn_mutuo(features_FILTER, sim_mat, k) 
            _, metrics  = evaluar_k_MNN_grafo(args.semilla, G_mutuo, s_min=3)
            results.append([sim_name, k, metrics["Q"], metrics["Porc_iso"], metrics["num_comptes"], metrics["num_comptes_small"], metrics["total_nodos"], metrics["num_nodos_iso"], metrics["num_comudes"]])
    #.-
    df = pd.DataFrame(results, columns=['sim_name','k','Q','Porc_iso','num_comptes', 'num_comptes_small', 'total_nodos', 'num_nodos_iso', 'num_comudes' ])
    ylim_Q   = (df["Q"].min()*0.995, df["Q"].max()*1.005)
    ylim_iso = (0, df["Porc_iso"].max()*100*1.05)
    k_optimos = {}
    for sim_name in df["sim_name"].unique():  ## Gráficos con similitud pearson y spearmean
        df_sim = df[df["sim_name"] == sim_name]
        #
        k_optimo = elegir_k_optimo(df_sim, args.ISO_UMBRAL, args.DELTA_Q_MAX)
        k_optimos[sim_name] = k_optimo
        #
        titulo = f"Métricas para distintos k con similitud {sim_name} ({args.nom_dataset})"
        archivo_evaluar_grafo = './_SALIDA/PDF/'+args.nom_dataset+'/Metricas_Varios_Grafos_'+sim_name+'.pdf'
        save_metricas_grafo(df_sim, archivo_evaluar_grafo, titulo, ylim_Q, ylim_iso, k_optimo)  
        #.-
        print(f"└─ {sim_name}: k óptimo sugerido = {k_optimo}")
    return k_optimos


#__________________________________________________
#
#       GRAFO QUE GUÍA LA BÚSQUEDA DEL NSGA-II EN LA LS
#.-_______________________________________________
def topk_cand_auto(tamanos):
    p75 = np.percentile(tamanos, 75)
    topk = np.ceil(0.25 * p75)
    topk = np.clip(topk, 5, 50)
    return topk
"""
top_nodes_por_comunidad: diccionario {comunidad: array de nodos} que guarda hasta TOPK_CAND nodos por comunidad, ordenados de mayor a menor grado.
"""
def grafo_guiador_LS(semilla, sim_mat, nom_dataset, k_optimo, features_FILTER, file_grafo_disco):
    print(f"Creando o leyendo grafo con k ={k_optimo} según corresponda para guiar la búsqueda ....")
    G_mutuo, partition, _ = cargar_o_crear_grafo_K_MNN_con_mejor_k(
        semilla, file_grafo_disco, features_FILTER, sim_mat, nom_dataset, k_optimo
    )
    if not partition:
        raise RuntimeError("partition vacío inesperadamente.")
    # Se asume que los IDs de nodo van de 0 a n_genes-1
    n_genes = int(G_mutuo.number_of_nodes())
    # --------------------------------------------------------------
    # Comunidad de cada nodo (vector indexable por ID; -1 = sin comunidad)
    # fromiter Crea un array de NumPy de una dimensión de los valores del dict.keys() (con los nodos de las comunidades en este caso)
    # --------------------------------------------------------------
    n_part = len(partition)
    nodos_array = np.fromiter(partition.keys(), dtype=np.int32, count=n_part)
    com_raw = np.fromiter(partition.values(), dtype=np.int32, count=n_part)
    comunidad_of_nodos = np.full(n_genes, -1, dtype=np.int32)
    comunidad_of_nodos[nodos_array] = com_raw
    # --------------------------------------------------------------
    # grado ponderado de cada nodo 
    # --------------------------------------------------------------
    deg_items = list(G_mutuo.degree(weight="weight"))
    nodes_deg = np.fromiter((n for n, _ in deg_items), dtype=np.int32, count=len(deg_items))
    vals_deg = np.fromiter((d for _, d in deg_items), dtype=np.float32, count=len(deg_items))
    grado_ponderado_all = np.zeros(n_genes, dtype=np.float32)
    grado_ponderado_all[nodes_deg] = vals_deg
    # --------------------------------------------------------------
    # Ordenar por comunidad y, dentro de cada una, por grado descendente
    # --------------------------------------------------------------
    degs_part = grado_ponderado_all[nodos_array]  
    sort_idx = np.lexsort((-degs_part, com_raw))  # ordena primero por comunidad y luego por -grado (el - es para orden de mayor a menor)
    com_s = com_raw[sort_idx]
    nod_s = nodos_array[sort_idx]
    # Límites de cada comunidad dentro de los arrays ordenados
    cut = np.flatnonzero(com_s[1:] != com_s[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int32), cut.astype(np.int32)))
    ends = np.concatenate((cut.astype(np.int32), np.array([com_s.size], dtype=np.int32)))
    
    # Comunidades únicas (ya ordenadas)
    inds_comunidades = com_s[starts]

    # --------------------------------------------------------------
    # top_nodes_por_comunidad: diccionario {comunidad: array de nodos} 
    # --------------------------------------------------------------
    TOPK_CAND = topk_cand_auto(ends - starts)
    top_nodes_por_comunidad = {
        int(com_s[s]): nod_s[s:min(e, s + TOPK_CAND)]
        for s, e in zip(starts, ends)
    }
    return (G_mutuo, comunidad_of_nodos, inds_comunidades, top_nodes_por_comunidad)

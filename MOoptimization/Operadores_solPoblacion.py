import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
"""
gene_initial_prob:
para cada una de las tam_sol columnas de una fila, se lanza una moneda cargada que cae en 1 con probabilidad gene_initial_prob y en 0 con probabilidad 1 − gene_initial_prob.
Se puede tomar como: "el porcentaje aproximado de genes que arrancarán activos en cada individuo".
"""
def initial_pop(rng, tam_pob, tam_sol, gene_initial_prob):
    mat = rng.random((tam_pob, tam_sol)) < gene_initial_prob # bool (N,d)
    vacias = ~mat.any(axis=1)
    if vacias.any(): # casi nunca debiera entrar
        filas = np.flatnonzero(vacias)
        mat[filas, rng.integers(0, tam_sol, size=filas.size)] = True
    return np.ascontiguousarray(mat)


"""
SELECTION por torneo binario canónico de NSGA-II (rank + crowding distance).
Produce 'lam' padres y los divide en dos sub-pools del mismo origen.
parents_a y parents_b provienen del mismo torneo binario sobre toda la población y solo se dividen para alimentar a crossover,
que empareja parents_a[i] con parents_b[i] para producir dos hijos por cruce.
Si lam es impar, parents_b queda con un elemento más.
"""
def rank_crowd_from_fronts(fitness, fronts, N):
    """Rank (índice de frente) y crowding por individuo, a partir de un sorting
    YA calculado. Permite un único NDS por generación."""
    rank = np.empty(N, dtype=np.int32)
    crowd = np.zeros(N, dtype=float)
    for i, fr in enumerate(fronts):
        rank[fr] = i
        cd = calc_crowding_distance(fitness[fr])
        crowd[fr] = np.nan_to_num(cd, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return rank, crowd

def selection(rng, population, fitness_pop, fronts, lam): #generar lam padres (no N)
    N = population.shape[0]
    # Torneo binario con rank+crowding ya calculados. Genera lam padres
    rank, crowd = rank_crowd_from_fronts(fitness_pop, fronts, N)
    idx1 = rng.integers(0, N, size=lam)  # rng.integers(0, N, size=N)
    idx2 = rng.integers(0, N, size=lam)  # rng.integers(0, N, size=N)
    mejor = (rank[idx1] < rank[idx2]) | ((rank[idx1] == rank[idx2]) & (crowd[idx1] > crowd[idx2]))
    parents_idx = np.where(mejor, idx1, idx2)
    pool = population[parents_idx]
    # Dividir el mating pool para alimentar a crossover en parejas (a[i], b[i]).
    # Si lam es impar, parents_b tiene un elemento más.
    mitad = lam//2 
    parents_a = pool[:mitad].astype(np.bool_, copy=False)
    parents_b = pool[mitad:].astype(np.bool_, copy=False)
    return parents_a, parents_b


"""
CROSSOVER Uniform canónico para NSGA-II:
- Con probabilidad crossover_rate: aplica uniform crossover (mezcla 50/50 por gen).
- Con probabilidad (1 - crossover_rate): los hijos son copias de los padres.
Produce lam hijos.
Ejemplo con crossover_rate=0.9:
    - 90% de las parejas hacen uniform crossover canónico (mezcla 50/50 por gen): Los hijos son una mezcla genuina de ambos padres.
    - 10% de las parejas no cruzan: los hijos son copias de los padres.
    - Diversidad de hijos: alta. Exploración real del espacio.
"""
def crossover(rng, parents_a, parents_b, crossover_rate, lam):
    n_a, n_b = parents_a.shape[0], parents_b.shape[0]
    # Caso borde: si uno de los grupos viene vacío, samplea del otro
    if n_a == 0 or n_b == 0:
        parent_pool = parents_a if n_a else parents_b
        idx = rng.integers(0, parent_pool.shape[0], size=lam)
        return np.ascontiguousarray(parent_pool[idx].astype(np.bool_, copy=False))
    n_genes = parents_a.shape[1]
    m = max(1, (lam + 1) // 2)  # nº de cruces (cada uno produce 2 hijos)
    idx_a = rng.integers(0, n_a, size=m)
    idx_b = rng.integers(0, n_b, size=m)
    p_a, p_b = parents_a[idx_a], parents_b[idx_b]

    # 1) Decidir qué parejas hacen crossover (uniforme por pareja)
    do_cx = rng.random(m) < crossover_rate  # (m,) bool

    # 2) Para las parejas que SÍ hacen crossover: máscara uniform 50/50 por gen.
    #    Para las que NO: máscara todo False 
    mask = np.zeros((m, n_genes), dtype=bool)
    if np.any(do_cx):
        mask[do_cx] = rng.random((int(do_cx.sum()), n_genes)) < 0.5
    children0 = np.where(mask, p_b, p_a)
    children1 = np.where(mask, p_a, p_b)
    pop_children = np.vstack([children0, children1])[:lam]
    return np.ascontiguousarray(pop_children.astype(np.bool_, copy=False))

"""
MUTATION estándar para representaciones binarias.
p_mut_ind : Probabilidad de que un individuo sea seleccionado para mutar.
p_mut_gene : Probabilidad por gen de ser flippeado dentro de un individuo seleccionado.
force_at_least_one_change : Si True, garantiza que un individuo seleccionado para mutar
    cambie efectivamente al menos un gen (evita que muestreos bajos de p_mut_gene produzcan "mutaciones nulas"). NO garantiza que el
    individuo resultante tenga al menos un gen activo; esa garantía la aporta ensure_non_empty_rows al final.
"""
def ensure_non_empty_rows(rng, pop):
    pop = np.asarray(pop, dtype=bool).copy()
    empty = pop.sum(axis=1) == 0
    if np.any(empty):
        cols = rng.integers(0, pop.shape[1], size=int(empty.sum()))
        pop[empty, cols] = True
    return pop

def mutation(rng, pop_children, p_mut_ind, p_mut_gene, force_at_least_one_change=True):
    pop_children = np.asarray(pop_children, dtype=bool)
    n_ind, d = pop_children.shape
    mutated = pop_children.copy()
    # Seleccionar individuos que mutan
    ind_mask = rng.random(n_ind) < p_mut_ind
    if np.any(ind_mask):
        # Seleccionar genes a mutar dentro de esos individuos
        bit_mask = rng.random((ind_mask.sum(), d)) < p_mut_gene
        if force_at_least_one_change and d > 0:
            empty_rows = ~bit_mask.any(axis=1)
            if np.any(empty_rows):
                cols = rng.integers(0, d, size=empty_rows.sum())
                bit_mask[empty_rows, cols] = True
        mutated[ind_mask] ^= bit_mask
    return ensure_non_empty_rows(rng, mutated) 


"""
SUPERVIVENCIA SELECTION.
Combina población actual con hijos, ordena por rank+crowding, y selecciona
los primeros N únicos. Si la unión no contiene N genotipos únicos
(convergencia masiva), completa permitiendo repeticiones para mantener
el tamaño N invariante; ese caso emite señal vía el flag interno.
population_sup : np.ndarray (N, n_genes) bool
fitness_sup : np.ndarray (N, 2) float
"""
def rank_crowding_full_order(F, nds):
    fronts = nds.do(F)
    ordered = []
    for front in fronts:
        front = np.asarray(front, dtype=int)
        cd = calc_crowding_distance(F[front])
        cd = np.nan_to_num(cd, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        # Dentro de cada frente: mayor crowding primero
        ordered.extend(front[np.argsort(cd)[::-1]].tolist())
    return np.asarray(ordered, dtype=int)

def supervivencia(population, pop_children, fitness_pop, fitness_pop_children, nds):
    union_pop = np.vstack((population, pop_children))
    F_union = np.vstack((fitness_pop, fitness_pop_children))
    idx_order = rank_crowding_full_order(F_union, nds) # permutación 0..2N-1 por prioridad
    N = population.shape[0]
    packed = np.packbits(union_pop.astype(np.bool_, copy=False), axis=1)
    packed_ordered = packed[idx_order] # genotipos en orden de prioridad
    # 1ª ocurrencia de cada genotipo único (np.unique con sort estable → primer índice)
    _, first_occ = np.unique(packed_ordered, axis=0, return_index=True)
    keep = np.sort(first_occ)  # posiciones en idx_order, en prioridad
    selected = idx_order[keep[:N]]
    n_unique = keep.size
    incluye_duplicados = n_unique < N
    if incluye_duplicados: # raro: rellenar con repetidos, en prioridad
        rest_mask = np.ones(idx_order.size, dtype=bool)
        rest_mask[keep] = False
        rest_positions = np.flatnonzero(rest_mask)
        selected = np.concatenate([selected, idx_order[rest_positions[:N - n_unique]]])
    return (np.ascontiguousarray(union_pop[selected].astype(np.bool_, copy=False)),
            F_union[selected].copy(), bool(incluye_duplicados))

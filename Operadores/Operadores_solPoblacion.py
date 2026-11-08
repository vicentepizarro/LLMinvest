import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# Generar una matriz aleatoria de probabilidades entre cero y uno y se activan los que son menores al umbral
# La matriz de probabilidades aleatorias es de tamaño (pop_size, num_genes)
def create_initial_population(pop_size, num_genes, gene_initial_prob):
    matrix_probabilities = np.random.rand(pop_size, num_genes)    
    population = matrix_probabilities < gene_initial_prob # Activar genes con probabilidad  menor al tope
    if num_genes:
        empty_rows = np.flatnonzero(~population.any(axis=1))
        if empty_rows.size:
            random_columns = np.random.randint(num_genes, size=empty_rows.size)
            population[empty_rows, random_columns] = True
    return population

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Selecciona los mejores individuos según su valor de fitnes y los deja en pop_initial
# En pop_remaining se dejan los resagados
def selection(population, fitness, percentage_select):
    total_indivs_select = int(population.shape[0] * percentage_select)  #número de individuos a seleccionar (los mejores)
    sorted_indices = np.argsort(fitness)[::-1]  #Ordenar índices por fitness en orden descendente 
    pop_initial = population[sorted_indices[:total_indivs_select]]   # Toma las primeras "total_indivs_select" filas de population 
    fitness_pop_initial = fitness[sorted_indices[:total_indivs_select]]  # con sus correspondientes fitnes
    pop_remaining = population[sorted_indices[total_indivs_select:]]  # toma el resto de las filas de population
    return pop_remaining, pop_initial, fitness_pop_initial  #los fitnes de mating_pool se obtienen después

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Para cada par de nuevos individuos (hijos) que se generarán, se seleccionan dos padres al azar.
# Un padre es de pop_initial y el  otro es de pop_remaining
# crossover_rate:  representa la probabilidad de que un gen específico de la descendencia sea tomado del primer padre 
#                            en lugar del segundo. Para esto se genera una máscara aleatoria con True si  el random aleatorio es < que el crossover_rate.
#                           Si crossover_rate es alto (cercano a 1), es más probable que los genes sean tomados de parent1. 
#                           Si es bajo (cercano a 0), es más probable que los genes sean tomados de parent2.
#                           Si es 0.5  asegura una mezcla balanceada de genes de ambos padres, promoviendo la diversidad genética en la población.
#  Se crean dos hijos: Cada hijo hereda el gen del primer padre si en la máscara existe u True, en caso contrario, hereda el gen del segundo padre
#       padre1:         [True, False, False, False], padre2: [False, True, False, True] y mask: [False, False, False, True]
#       hijo1 sería:    [False  True False False]
def crossover(pop_remaining, pop_initial, crossover_rate):
    new_population = np.empty((pop_remaining.shape[0], pop_remaining.shape[1]), dtype=pop_remaining.dtype)
    num_descendencia = 0
    while num_descendencia < pop_remaining.shape[0]: # descendencia del mismo tamaño que pop_remaining
        # Probabilidades del gen descendiente (si se hereda  del padre 1 en vez del padre 2 )
        parent1 = pop_initial[np.random.randint(pop_initial.shape[0])] #padre de pop_initial
        parent2 = pop_remaining[np.random.randint(pop_remaining.shape[0])] #padre de pop_remaining
        prob_desc = np.random.rand(pop_remaining.shape[1]) 
        child1 = np.where(prob_desc< crossover_rate, parent1, parent2)
        child2 = np.where(prob_desc< crossover_rate, parent2, parent1)
        if child1.size and not np.any(child1):
            child1[np.random.randint(child1.size)] = True
        if child2.size and not np.any(child2):
            child2[np.random.randint(child2.size)] = True
        new_population[num_descendencia] = child1 # descendencia 0, 
        if (num_descendencia + 1) < pop_remaining.shape[0]: # asegura que el tamaño de la nueva pobla sea igual a pop_remaining
            num_descendencia = num_descendencia + 1
            new_population[num_descendencia] = child2  # descendencia 1,
        num_descendencia = num_descendencia + 1 # para la siguiente descendencia
    return new_population[:pop_remaining.shape[0]] #
    
# ---------------------------------------------------------------------------------------------------------------------
# mutation_rate: Representa la probabilidad de que un individuo sea nutado o no.
#                         Se genera un array(tam_pop_chilren,) con valores aleatorios. 
#                         Los que son < a mutation_rate son los individuos que se mutan
# gene_mutation_rate: Representa la probabilidad de que un gen de un  un individuo sea mutado.
#                         Se genera una matriz(tam_pop_chilren, num_genes) con valores aleatorios. 
#                         Los que son < a gene_mutation_rate son los genes que se mutan
# np.logical_xor(a.b): Cambia los genes del individuo a si en b tiene un True
#                               [True, True, False, False]  y   [False, False, False, True]     devuelve   [True, True, False, True]
#                               [True, True, False, False]  y   [True, True, False, True]      devuelve   [False, False, False, True]
def mutation(pop_children,  mutation_rate, gene_mutation_rate): # de tamaño igual a mating_pool = pop_children
    # Índices de los individuos que serán mutados
    inds_individuos_a_mutar = np.where(np.random.rand(pop_children.shape[0]) < mutation_rate)[0] 
    # Matriz de genes a mutar. Los con valores True se mutan
    random_matrix_genes_a_mutar = np.random.rand(pop_children.shape[0], pop_children.shape[1]) <  gene_mutation_rate
    random_matrix_genes_a_mutar = random_matrix_genes_a_mutar[inds_individuos_a_mutar,:] # se dejan sólo las filas con los individuoa a mutar
    # Se cambia el gen de pop_children[inds_individuos_a_mutar] cuando en random_matrix_genes_a_mutar es True
    mutated_pop_children = np.logical_xor(pop_children[inds_individuos_a_mutar], random_matrix_genes_a_mutar)
    pop_children[inds_individuos_a_mutar] = mutated_pop_children
    if pop_children.shape[1]:
        empty_rows = np.flatnonzero(~pop_children.any(axis=1))
        if empty_rows.size:
            random_columns = np.random.randint(pop_children.shape[1], size=empty_rows.size)
            pop_children[empty_rows, random_columns] = True
    return pop_children



import numpy as np
from math import inf
from EVALfitness_MonoObj.Funciones_Optimization import fitness_function_RL
# Seleccionar un gen aleatorio sin preocuparse si está seleccionado o no
def gen_aleatorio(curr_sol_bin):
    ind_gen_rand = np.random.randint(curr_sol_bin.shape[0])  # es lo mismo: np.random.choice(np.where(curr_sol_bin)[0])
    return ind_gen_rand                                                                        # porque curr_sol_bin es binario
    
# Seleccionar un rango (bloque) dentro de la solución
def rango_solucion (curr_sol_bin, size_bloque):  
        ini = np.random.randint(curr_sol_bin.shape[0] - size_bloque)  #El inidice máximo no lo va a retornar
        fin = ini + size_bloque - 1
        return ini, fin
        
# Seleccionar uno o varios genes aleatorios distintos (replace=False) según cuantos_genes_sel
def genes_aleatorios(curr_sol_bin, cuantos_genes_sel):
    inds_genes = np.random.choice(curr_sol_bin.shape[0], cuantos_genes_sel, replace=False) #retorna un array
    if cuantos_genes_sel == 1:
        return inds_genes[0]  # devuelve un gen
    else:
        return inds_genes  # devuelve el arreglo de genes    
        
# Seleccionar un gen que NO está seleccionado en curr_sol
def gen_alreatorio_notselected(curr_sol_bin):
    inds_gen_false = np.where(~curr_sol_bin)[0]  #todos los índices de los genes False (no seleccionados)
    if inds_gen_false.size == 0 and curr_sol_bin.size:
        ind_gen = np.random.randint(curr_sol_bin.shape[0])
        curr_sol_bin[ind_gen] = False
        return ind_gen
    return np.random.choice(inds_gen_false)  # retorna uno (no es necesario usar:  inds_gen_false,  1, replace=False)

# Seleccionar un gen que esté seleccionado en curr_sol
def gen_alreatorio_selected(curr_sol_bin):
    inds_gen_true = np.where(curr_sol_bin)[0]  # todos los índices de los genes True
    if inds_gen_true.size == 0 and curr_sol_bin.size:
        ind_gen = np.random.randint(curr_sol_bin.shape[0])
        curr_sol_bin[ind_gen] = True
        return ind_gen
    return np.random.choice(inds_gen_true)  # retorna uno (no es necesario usar:  inds_gen_false,  1, replace=False)

# Seleccionar genes con baja corr, alta corr, etc. según corresponda   que NO estén seleccionados en curr_sol
def gen_inarray_alreatorio_notselected(indices_genes, curr_sol_bin):  
    inds_gen_true = np.where(curr_sol_bin)[0] # genes activados
    inds_notin_curr_sol = np.setdiff1d(indices_genes, inds_gen_true)  #  indices_genes que no estén seleccionados (en inds_gen_true)
    if inds_notin_curr_sol.shape[0]>0:
        ind_gen_notsel = np.random.choice(inds_notin_curr_sol) # uno gen aleatorio de los encontrados y que no estén seleccionados
    else:
        ind_gen_notsel = inf  # retorna "inf" cuando no existen indices_genes que estén seleccionados 
    return ind_gen_notsel
                
# Generar solución inicial
def generar_solucion_ini(num_features_tot, gene_initial_prob):    #tesis de Cubillos con 208 logra un rendimiento 0.875
    curr_sol_bin = np.random.choice([False, True], size=num_features_tot, p=[1-gene_initial_prob, gene_initial_prob])
    if num_features_tot and not np.any(curr_sol_bin):
        curr_sol_bin[np.random.randint(num_features_tot)] = True
    return curr_sol_bin  #Devuelve un array binario con shape: (num_features_tot,)
    
 # Activar o desactivar UN solo gen (tiro la moneda  entre los seleccionados y no seleccionados)
def mutacion_simple(curr_sol_bin):
    if curr_sol_bin.size == 0:
        return curr_sol_bin
    if not np.any(curr_sol_bin):
        curr_sol_bin[np.random.randint(curr_sol_bin.shape[0])] = True
        return curr_sol_bin
    #gen_sel = gen_aleatorio(curr_sol_bin)
    moneda = np.random.randint(2)
    if moneda == 1:
        gen_sel= gen_alreatorio_selected(curr_sol_bin)            # gen selecccionado
    else:
        gen_sel = gen_alreatorio_notselected(curr_sol_bin)    # gen no seleccionado
    curr_sol_bin[gen_sel] = not curr_sol_bin[gen_sel] #hace la mutación
    if not np.any(curr_sol_bin):
        curr_sol_bin[gen_sel] = True
    return curr_sol_bin
    
# Eliminar un gen cualquiera seleccionado  (sin considerar correlación)
def eliminar(curr_sol_bin):
    if np.sum(curr_sol_bin) <= 1:
        return curr_sol_bin  # Una solución debe conservar al menos un gen
    gen_sel = gen_alreatorio_selected(curr_sol_bin)  # # toma un gen seleccionado
    curr_sol_bin[gen_sel] = False # elimino nuevo gen
    return curr_sol_bin  #np.where(features_sel_bin)[0]
    
# Agregar un gen cualquiera no seleccionado  (sin considerar correlación)
def agregar(curr_sol_bin):
    if np.sum(curr_sol_bin) >= curr_sol_bin.shape[0]:
        return curr_sol_bin  # No se puede agregar
    ind_gen_false = gen_alreatorio_notselected(curr_sol_bin) # toma un gen no seleccionado
    curr_sol_bin[ind_gen_false] = True  # se agrega
    return curr_sol_bin
    
 # Intercambiar un gen seleccionado por otro no seleccionado (sin considerar correlación)
def intercambiar(curr_sol_bin): 
    if curr_sol_bin.size == 0:
        return curr_sol_bin
    ind_gen_true = gen_alreatorio_selected(curr_sol_bin)            # gen selecccionado
    ind_gen_false = gen_alreatorio_notselected(curr_sol_bin)    # gen no seleccionado
    curr_sol_bin[ind_gen_true] = False   # desactivar 
    curr_sol_bin[ind_gen_false] = True   # activar
    return curr_sol_bin

# Invertir un bloque de genes
def invertir_rango(curr_sol_bin, args): # args.max_size_bloque=3
        size_rango = args.max_size_bloque # np.random.randint(3, args.max_size_bloque) 
        ini, fin = rango_solucion(curr_sol_bin,size_rango)  # obtener rango (o bloque)
        curr_sol_bin[ini:fin+1] = curr_sol_bin[ini:fin+1][::-1]  #invertit
        return curr_sol_bin
    
# Limpia bloques dejando todos los elementos inacticos
# falta barrer


# Activar o desactivar un rango der genes de tamaño 4 inicialmente
def mutar_bloque(curr_sol_bin, args):
        size_rango = args.max_size_bloque + 1  # np.random.randint(3, args.max_size_bloque)
        ini, fin = rango_solucion(curr_sol_bin, size_rango)  # obtener rango (o bloque)
        curr_sol_bin[ini:fin+1] = curr_sol_bin[ini:fin+1] = curr_sol_bin[ini:fin+1] ^ 1 #cambiar 0 por 1 y viciversa
        return curr_sol_bin

# Mutar genes aletorios según parametro porc_mutacion (= 0.002)
def mutacion_aleatoria(curr_sol_bin, args):
        cuantos = int(args.porc_mutacion*curr_sol_bin.shape[0])  # 3079*0.002 = 6 genes
        inds_a_mutar = genes_aleatorios(curr_sol_bin, cuantos)
        curr_sol_bin[inds_a_mutar] = curr_sol_bin[inds_a_mutar] ^ 1   # #cambiar 0 por 1 y viciversa
        return curr_sol_bin
 
#    -----------------------------------------------------------
#                 Operadores Búsqueda Local
#   ------------------------------------------------------------

# Agregar un gen correlacionado no seleccionado 
def agregar_genbacorr(curr_sol_bin, genes_baja_corr): 
    if np.sum(curr_sol_bin) >= curr_sol_bin.shape[0]:
        return curr_sol_bin  # No se puede agregar
    gen_notsel = gen_inarray_alreatorio_notselected(genes_baja_corr, curr_sol_bin) # tomo uno con baja correlación y no seleccionado
    if gen_notsel != inf:  # valor inf es cuando exiten genes con baja coorrs pero todos ya están seleccionados 
        curr_sol_bin[gen_notsel] = True # agrego nuevo gen con baja correlación
        return curr_sol_bin 
    else:
        return agregar(curr_sol_bin) #sin correlación

# Seleccionar genes altamente correlacionados con ind_gen 
def genes_corrs_con_gen(ind_gen, corr_matriz):
    genes_corrs = corr_matriz[ind_gen,:] # obtener genes correlaciones con ind_gen   
    genes_corr = np.where(genes_corrs >= 0)[0]  # (genes_corrs >= 0) es un array numpy binario
    return  genes_corr # devuelve un array unidimensional (caso de np.where(..) sólo con condición)

# Intercambiar un gen seleccionado con otro que esté altamente correlacionado y no seleccionado
def intercambiar_con_gencorr(curr_sol_bin, corr_matriz): 
    if curr_sol_bin.size == 0:
        return curr_sol_bin
    if not np.any(curr_sol_bin):
        curr_sol_bin[np.random.randint(curr_sol_bin.shape[0])] = True
    if np.all(curr_sol_bin):
        return curr_sol_bin
    gen_sel = gen_alreatorio_selected(curr_sol_bin)  # busca gen que esté seleccionado
    flag = 1 # inicializar intercambio no realizado
    genes_corrs = genes_corrs_con_gen(gen_sel, corr_matriz) # lista de genes correlacionados con gen_sel
    if genes_corrs.any():
        gen_notsel = gen_inarray_alreatorio_notselected(genes_corrs, curr_sol_bin) # de esta lista,  que no esté seleccionado
        if gen_notsel != inf:  # valor inf es cuando exiten corrs pero todos ya están seleccionados 
            curr_sol_bin[gen_sel] = False  # desactivo
            curr_sol_bin[gen_notsel] = True  # activo 
            flag = 0 # intecambio OK
    if flag == 1: # no encontró genes corrs o si existen  ya están seleccionados
        all_indices = np.arange(curr_sol_bin.shape[0]) 
        gen_notsel = gen_inarray_alreatorio_notselected(all_indices, curr_sol_bin) ## aquí siempre encuentra uno para intercambiar
        curr_sol_bin[gen_sel] = False # desactivo
        curr_sol_bin[gen_notsel] = True # activo 
    return curr_sol_bin # esto no: np.where(curr_sol_bin)[0] ya que no necesitamos retornar los índices
    
    
    
# Operador de búsqueda local ESTRATEGIA 1
def local_search_E1(sol_LS, costo_sol_LS, corr_matriz, num_evaluaciones,X, y, args): 
    i = 0
    while i < args.num_vecinos_LS:  #Criterio de detención de LS
            s1= intercambiar_con_gencorr(sol_LS, corr_matriz)
            s1_costo= fitness_function_RL(s1, X, y, 'min',args)
            num_evaluaciones = num_evaluaciones + 1
            if  (s1_costo < costo_sol_LS) :  # mejora la solución que entra (sol = xprim de VNS)
                    sol_LS, costo_sol_LS = s1, s1_costo
                    break          # Estrategia    
            else:
                    i = i + 1       # Estrategia
    return  sol_LS, costo_sol_LS, num_evaluaciones  






# Operador de búsqueda local ESTRATEGIA 2
def local_search_E2(sol_LS, costo_sol_LS, corr_matriz, num_evaluaciones,X, y, args):  
    i = 0
    while i < args.num_vecinos_LS:  #Criterio de detención de LS
            s1= intercambiar_con_gencorr(sol_LS, corr_matriz)
            s1_costo= fitness_function_RL(s1, X, y, 'min', args)
            num_evaluaciones = num_evaluaciones + 1
            if  (s1_costo < costo_sol_LS) :  # mejora la solución que entra (sol = xprim de VNS)
                    sol_LS, costo_sol_LS = s1, s1_costo
            i = i + 1       # Estrategia
    return  sol_LS, costo_sol_LS, num_evaluaciones  

# Operador de búsqueda local ESTRATEGIA 2
def local_search_sinLS(sol_LS, costo_sol_LS,  num_evaluaciones,X, y, args):  
    i = 0
    while i < args.num_vecinos_LS:  #Criterio de detención de LS
            s1= intercambiar(sol_LS) 
            s1_costo= fitness_function_RL(s1, X, y, 'min', args)
            num_evaluaciones = num_evaluaciones + 1
            if  (s1_costo < costo_sol_LS) :  # mejora la solución que entra (sol = xprim de VNS)
                    sol_LS, costo_sol_LS = s1, s1_costo
                    break          # Estrategia    
            else:
                    i = i + 1       # Estrategia
    return  sol_LS, costo_sol_LS, num_evaluaciones  

#  ---------------------------------------------------------------------------
#  OPERADORES  CON y SIN BÚSQUEDA LOCAL
# ----------------------------------------------------------------------------
prob_operadores = np.asarray([1/4, 1/4, 1/4, 1/4])  # voy a probar con la misma probabilidad  para cada uno
operadoresSA_conLS = np.array([
            mutacion_simple,      # activar o desactivar UN solo gen, 
            eliminar,
            agregar_genbacorr, 
            intercambiar  # activar uno y desactivar el otro (intercambio simple sin correlación
            #invertir_rango,   #inicialmente tamaño del rango es 3
            #mutar_bloque,   # #inicialmente tamaño del bloque es 4
            #mutacion_aleatoria     # mutar genes aletorios segpun parametro porc_mutacion (= 0.002 = 6 genes)
            ])
            
operadoresSA_sinLS = np.array([
            mutacion_simple,      # activar o desactivar UN solo gen, 
            eliminar,
            agregar, 
            intercambiar  # activar uno y desactivar el otro (intercambio simple sin correlación
            ])
#  ---------------------------------------------------------------------------
#  OPERADORES y VECINDARIOS para SA y VNS SIN BÚSQUEDA LOCAL
# ----------------------------------------------------------------------------

# Vecindario para VNS
vecindarioVNS_sinLS = np.array([
            mutacion_simple,      # activar o desactivar UN solo gen, 
            eliminar,
            agregar, 
            intercambiar  # activar uno y desactivar el otro (intercambio simple sin correlación
            ])

vecindarioVNS_conLS = np.array([
            mutacion_simple,      # activar o desactivar UN solo gen, 
            eliminar,
            agregar_genbacorr, 
            intercambiar  # activar uno y desactivar el otro (intercambio simple sin correlación
#            invertir_rango,   #inicialmente tamaño del rango es 3
#            mutar_bloque,   # #inicialmente tamaño del bloque es 4
#            mutacion_aleatoria     # mutar genes aletorios segpun parametro porc_mutacion (= 0.002 = 6 genes)
            ])


# Seleccionar el operador para perturbación
def roulette_wheel(prob_operador):
    # probabilidades acumulativas: Ejemplo: [1/3, 2/3, 1] de [1/3, 1/3, 1/3]
    cumulative_sum = np.cumsum(prob_operador) 
    prob_detencion = np.random.rand() # valor float desde 0.0 a 1.0 (no inclusive el 1)
    return np.searchsorted(cumulative_sum, prob_detencion) # Busca la primera posición (0, 1 o 2) en cumulative_sum [1/3, 2/3, 1]  
                                                           # que es mayor o igual a prob_detencion.
# 
def apply_operador(operador_run, curr_sol_bin):
    # El primer parámetro de apply_vecindadNK es el nombre de la función a ejecutar
    return operador_run(curr_sol_bin) # aquí ejecuta operador

def apply_operador_2(operador_run, curr_sol_bin, genes_baja_corr):
    # El primer parámetro de apply_vecindadNK es el nombre de la función a ejecutar
    if operador_run == vecindarioVNS_conLS[2]:
        return operador_run(curr_sol_bin, genes_baja_corr) # aquí ejecuta operador
    else:
        return operador_run(curr_sol_bin) # aquí ejecuta operador
         

def run_operador_conLS(curr_sol_bin, genes_baja_corr):
    ind_operador = roulette_wheel(prob_operadores) # recibe el índice del operador a ejecutar
    if ind_operador == 2:
        return apply_operador_2( operadoresSA_conLS[ind_operador], curr_sol_bin, genes_baja_corr)
    else:
        return apply_operador( operadoresSA_conLS[ind_operador], curr_sol_bin)

def run_operador_sinLS(curr_sol_bin):
    ind_operador = roulette_wheel(prob_operadores) # recibe el índice del operador a ejecutar
    return apply_operador( operadoresSA_sinLS[ind_operador], curr_sol_bin)
            
def promedios_movimiento_costo(costos, window_size):
    return np.convolve(costos, np.ones(window_size)/window_size, mode='valid')
    
def filtrar_correlaciones(corr_matriz, umbral_correlacion): 
    matriz_mask =  np.abs(corr_matriz) > umbral_correlacion  # mask  es matriz binaria con valores True los que cumplen la condición
    corr_matriz = np.where(matriz_mask, corr_matriz, -1) # Aplicar condición (matriz mask) a corr_matriz
                                                                                                    # caso np.where donde devuelve valores (no índices)
    np.fill_diagonal(corr_matriz, -1)
    return corr_matriz


  
#==================================================================
#  FUNCIONES USADAS EN LOS OPERADORES
#     -´seleccionar índice aleatorio de un array            : 
#                 np.random.randint(0, len(array))  => para recuperar su valor hago: array[indice_aleatorio]
#     - seleccionar un elemento aleatorio  de un array : 
#                 valor_aleatorio = np.random.choice(array)  => indice_aleatorio = np.where(array == valor_aleatorio)[0][0]

también esten en inds_gen_true se hace: np.intersect1d(indices_genes_corr, inds_gen_true)
# ---------------------------------------

x = x0 ; /∗ Generate the initial solution ∗/

while num_evaluaciones < args.num_eval_max: #criterio de detención
        
        for k in  range(k_max): # largo del vecindarioVNS
        
            if  num_evaluaciones > args.num_eval_max:
                print(f"En iteracion = {k} alcanzó número MAX de evaluaciones = {num_evaluaciones}")
                break           

            # (1): SHAKING: Generar solución aleatoria  x' (del k-ésimo vecindario)
            # ---------------------------------------------------------------------------------
            x_sol_bin_prima = neighborhood_shaking(curr_sol_bin, k, dataX.shape[1], corr_matriz_filtrada, genes_baja_corr) 
           
            # (2) LOCAL SEARCH: ¿ SIEMPRE SALE UNA SOLUCIÓN x'' MEJOR QUE x' ?????
            # ---------------------------------------------------------------------------------
            x_sol_bin_dobprima, costo_sol_dobprima, num_evaluaciones = local_search(x_sol_bin_prima, l_max, dataX.shape[1],corr_matriz_filtrada, genes_baja_corr, num_evaluaciones, X_train_scaled, y_train, rkf)
            
            # (3): MOVIMIENTO: evaluar curr_sol y x_sol_bin_doble_prima (x y x'')
            # -----------------------------------------------------------------------
            if costo_sol_dobprima < curr_costo:  #SINO => sigue con el próximo vecindario Nk
                #
                # PREGUNTA: ¿REINICIO??? (comenzar nuevamente a recorrer los vecindarios NK) 
                #                         
                curr_sol_bin, curr_costo = x_sol_bin_dobprima, costo_sol_dobprima
                array_costo, current_size = io.add_element(array_costo, curr_costo, current_size)
                
                print(f"COSTO MEJORADO = {curr_costo:.6f}.      Shape array costo = {array_costo.shape},    current_size={current_size}")
                
                break  # reiniciar el ciclo for del vecindario


# (a) Generar solución inicial: ALTO número de FS no alcanza a explorar números bajos
#                               BAJO número de FS no alcanza a explorar números más altos
#                               ¡Primero debemos definir qué es ALTO y qué es BAJO número de FS!
#                               En el ejemplo N=22000 ==> ¿ bajos= hasta 1/3, medidos = [1/3, 2/3]; alto = sobre 2/3 ?
#                                                                                       7000  14000
# SS para N = 22000: 
# SS con K = 200 (¿fijo?): 
# ¿ Será mejor DEFINIR cuantos genes se seleccionarán al menos inicialmente?

# ----------------------------------------------------------
#   DEFINICIÓN DE VECINDADES para shaking y Local Searh
# -----------------------------------------------------------
#                                                        0  0  1   1  0 1  0  0  0  1 0
#     Invertir_rango(1, 5)  =>          *  1   0  1  1  0  *   *  *  *  *    => mantiene el número de FS
#     mutacion_simple (2)  =>       *  *   0  *   *  *  *  *  *  *  * => NO mantiene el número de FS (es un agregar o eliminar un gen)
# vecindarioNK: 
#        invertir_rango,   #un bloque lo invierte
#        perturbar  ,     # varios cambios  de intercambios simples: invertir_simple, intercambiar, etc.
#        intercambiar,  #activar uno y desactivar el otro),  
#        agregar_genbacorr,   # Sube  en 1 el num de FS
#        agregar,     # Sube  en 1 el num de FS
#        eliminar,     # baja  en 1 el num de FS
#        mutacion_simple.  # activar o desactivar UN solo gen, 
 vecindarioNK = np.array([intercambiar_con_gencorr,  #se mantiene el num de FS
                                                intercambiar,   #se mantiene el num de FS. Se activa/desactiva DOS genes
                                                agregar_genbacorr,  #Sube  en 1 el num de FS
                                                agregar,   #Sube  en 1 el num de FS
                                                eliminar   #Elimina  en 1 el num de FS
                                                ])
        - intercambiar (activar uno y desactivar el otro),  
        - inversion_rango,   #un bloque lo invierte
        - perturbar  ,     # varios cambios  de intercambios simples: invertir_simple, intercambiar, etc.
        - agregar_genbacorr,   # Sube  en 1 el num de FS
        - agregar,     # Sube  en 1 el num de FS
        - eliminar.     # baja  en 1 el num de FS
        - invertir_simple # (o mutacion_simple): activar o desactivar UN solo gen, 

# OPERADORES: 
#   
#                                                        0  0  1   1  0 1  0  0  0  1 0
#     Invertir_rango(1, 5)  =>          *  1   0  1  1  0  *   *  *  *  *    => mantiene el número de FS
#     mutacion_simple (2)  =>       *  *   0  *   *  *  *  *  *  *  * => NO mantiene el número de FS (es un agregar o eliminar un gen)
# vecindarioVNS: 
        inversion_rango,   #un bloque lo invierte
        perturbar  ,     # varios cambios  de intercambios simples: invertir_simple, intercambiar, etc.
        intercambiar,  #activar uno y desactivar el otro),  
        agregar_genbacorr,   # Sube  en 1 el num de FS
        agregar,     # Sube  en 1 el num de FS
        eliminar,     # baja  en 1 el num de FS
        mutacion_simple.  # activar o desactivar UN solo gen, 
        
# vecindarioLS:   ==>  LA BUSQUEDA LOCAL INCLUYE EVALUCIÓN
       - intercambiar_con_gencorr   # le mantiene el num de FS
       
# OTROS  pueden ser (crircular o no dependiendo del caso):
      - barrer bloque (se mueve el bloque a una posición y donde estaba quedan desactivados)
      - contrario a barrer bloque (se mueve el bloque a una posición y donde estaba quedan activados)
      - rotar array  n posiciones de manera circular
      - Duplicar un bloque (copiarlo tal cual desde un índice aleatorio)
      - intercambiar dos bloques (tal como están)
      - intercambiar dos bloques (mutando_simple uno de ellos)
      - intercambiar dos bloques (mutando_simple los dos bloques)
      

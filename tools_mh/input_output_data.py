import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------
#           LECTURAS DATASETS de CUMIDA: columnas: genes y filas: muestras
# ----------------------------------
def leer_dataset_cumida_2clases(nom_file_in, sep_data, target_pos, target_neg):
    df = pd.read_csv(nom_file_in, sep=sep_data)   
    nom_col_class =  df.columns[1]
    df[nom_col_class] = df[nom_col_class].str.strip()
    lis_genes =  np.array(df.columns[2:])
    Y = df[nom_col_class].to_numpy()
    X = df.iloc[:, 2:].to_numpy()
    Y[Y == target_pos] = True  # con tumor
    Y[Y == target_neg] = False # sin tumor (o normal)
    Y = Y.astype(bool) # Y.astype(np.uint8) ocupa solo un byte por elemento
    return lis_genes, X, Y

def leer_dataset_cumida_2clases_2(nom_file_in, sep_data, target_pos, target_neg):
    df = pd.read_csv(nom_file_in, sep=sep_data)   
    nom_col_class =  df.columns[-1]
    df[nom_col_class] = df[nom_col_class].str.strip()
    lis_genes =  np.array(df.columns[1:-1])
    Y = df[nom_col_class].to_numpy()
    X = df.iloc[:, 1:-1].to_numpy()
    Y[Y == target_pos] = True  # con tumor
    Y[Y == target_neg] = False # sin tumor (o normal)
    Y = Y.astype(bool) # Y.astype(np.uint8) ocupa solo un byte por elemento
    return lis_genes, X, Y

def leer_matriz(nom_file_matriz):
    X = np.loadtxt(nom_file_matriz,  delimiter=args.sep,  dtype=np.float)

#
# Lee archivo con datos enteros y una columna 
# para almacenar en un array entero unidimensional 
def leer_unidim_dataint(nom_file_arrayint_unidim):
    array_unidim = np.loadtxt(nom_file_arrayint_unidim, dtype=np.int16)
    return array_unidim


#
# matriz numply que se agrega por fila
def add_row(matriz, row, current_size, eval_max):
    if current_size < matriz.shape[0]:
        matriz[current_size, :] = row
        current_size = current_size + 1
    else:
        raise ValueError(f"Se ha alcanzado el número máximo {eval_max} de filas permitido, current_size={current_size}" )
    return matriz, current_size
    
    
 # Función para agregar elementos a un array inteligente
def add_element(array_unidim, elemento, current_size):
    if current_size >= array_unidim.shape[0]:
        new_size = array_unidim.shape[0] * 2  # para Redimensionar
        new_array = np.empty(new_size)
        new_array[:current_size] = array_unidim
        array_unidim = new_array
    array_unidim[current_size] = elemento
    return array_unidim, current_size + 1

def leer_data_filtrado(args, param):
    X = np.loadtxt(param.data_X_FS_filter, delimiter=args.sep,  dtype=np.float64)
    Y_num = np.loadtxt(param.data_y_FS_filter, delimiter=args.sep,  dtype=np.uint8)
    features = np.loadtxt(param.data_FS_filter, delimiter=args.sep, dtype=str)
    return features, X, Y_num


# --------------------------------------------------
#  SALIDAS: GRAFICOS
# ----------------------------------------------
#
# Convergencia de la MH
#----------------------------------------
def curva_convergencia_VNS(array_costo,nom_file_graf, tit, num_eval_max):
    evaluaciones = np.arange(1, array_costo.shape[0] + 1)
    costs = array_costo[:, 1]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(evaluaciones, costs, '-', label='Costo de la solución',  linewidth=0.7)  #'o-'
    ax.set_xlabel('Numero de Evaluaciones', fontsize=12)
    ax.set_ylabel('Costo de la Solución', fontsize=12)
    titulo = tit
    ax.set_title(titulo + " (%s evaluaciones)" % (num_eval_max), fontsize=14)
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=10) 
    fig.savefig(nom_file_graf, dpi = 1080)
    #plt.clf()
    return

def curva_FS_VNS(array_costo,nom_file_graf, tit, num_eval_max):
    evaluaciones = np.arange(1, array_costo.shape[0] + 1)
    num_genes_selected = array_costo[:, 2]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(evaluaciones, num_genes_selected, '-',   linewidth=0.7)  #'o-'
    ax.set_xlabel('Numero de Evaluaciones', fontsize=12)
    ax.set_ylabel('Número de Genes Seleccionados', fontsize=12)   
    titulo = tit
    ax.set_title(titulo + " (%s evaluaciones)" % (num_eval_max), fontsize=14)
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=10)     
    fig.savefig(nom_file_graf, dpi = 1080)
    #plt.clf()
    return

#boxplot para visualizar la variabilidad y tendencia central del rendimiento.


def curva_convergencia(array_unidim_values,nom_file_graf, tit, num_eval_max):
    x = np.arange(1, array_unidim_values.shape[0] + 1)
    fig, ax = plt.subplots()
    ax.plot(x, array_unidim_values, '-')  #'o-'
    ax.set_xlabel('Numero de Evaluaciones')
    ax.set_ylabel('Costo/Calidad Fitness')
    titulo = tit
    ax.set_title(titulo + " (%s evaluaciones)" % (num_eval_max), fontsize=13)
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.savefig(nom_file_graf, dpi = 1080)
    plt.clf()
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def grafico_matriz_correlacion(matriz_correlacion):
    plt.figure(figsize=(20, 20))
    #cmap = sns.diverging_palette(220, 20, as_cmap=True) #personalizar los colores
    #fig, ax = plt.subplots()
    #fig.set_size_inches(15,15)
    sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    #sns.heatmap(matriz_correlacion, cmap=cmap)
    fig.savefig('./data/graficos/correlation_matrix.png', dpi = 1080)

def grafico_correlacion_con_target(X, correlaciones, umbral):
    #indices = np.arange(0,X.shape[1],1,dtype=np.int32)
    indices_ordenados = np.argsort(np.abs(correlaciones[:, 1]))[::-1]  # orden descendente
    correlaciones_ordenados = np.abs(correlaciones[indices_ordenados, 1])
    plt.figure(figsize=(10, 5))
    #plt.bar(correlaciones[indices_ordenados, 0], correlaciones[indices_ordenados, 1]) #, color='skyblue'
    #plt.bar(correlaciones[indices, 0], np.abs(correlaciones[indices, 1])) #, color='skyblue'
    plt.bar(np.arange(correlaciones_ordenados.shape[0]),correlaciones_ordenados, color='b', align='center')
    plt.xlabel('Características')
    plt.title(f'Correlación de Punto Biserial con Target - UMBRAL={umbral}')
    plt.ylabel('Correlación de Punto Biserial')
    plt.axhline(y=umbral, color='r', linestyle='--')
    plt.savefig('./data/graficos/correlation_con_target.png', dpi = 1080)
    plt.show() 

#
def grafico_informacion_mutua(X, mutual_info, umbral):
    indices_ordenados = np.argsort(mutual_info)[::-1]  # De mayor a menor
    mutual_info_ordenados = mutual_info[indices_ordenados]
    #indices = np.arange(0,X.shape[1],1,dtype=np.int32)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(mutual_info_ordenados.shape[0]),mutual_info_ordenados, color='b', align='center')
    #plt.xticks(range(X.shape[1]), np.arange(X.shape[1])[indices_ordenados], rotation=90) #, fontsize=12
    #plt.yticks(fontsize=12)
    plt.ylabel('Información Mutua')
    plt.xlabel('Características')
    #plt.title('MI entre las Características y el Objetivo ')    
    plt.title(f'MI entre las Características y el Objetivo -  UMBRAL={umbral}')
    plt.axhline(y=umbral, color='r', linestyle='--')
    plt.savefig('./data/graficos/informacion_mutua.png', dpi = 1080)
    plt.show()    

    
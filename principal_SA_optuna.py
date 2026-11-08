# 
# -------------------
import optuna
import numpy as np
import logging
from datetime import datetime
from tools_mh import input_output_data as io
from Filtros  import FS_metodos_filtros as filtros
from FS_optimizador_SA import simulated_annealing_optuna as sa
from Operadores import Operadores_solUnica as op
import time
import multiprocessing
import sys
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, plot_slice, plot_param_importances, plot_edf, plot_rank, plot_timeline
from optuna.trial import TrialState
from optuna.samplers import TPESampler, RandomSampler, GridSampler  # ver: https://datagy.io/python-optuna/
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import argparse
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


def obtener_parametros():
    par = argparse.ArgumentParser()
    par.add_argument('--num_eval_max', help="número máximo de evaluación a la FO", type=int, default=100000)
    par.add_argument('--isLS', help="Algoritmo con LS o sin LS", type=str, default='si')
    return par.parse_args()


def logging_callback(study, trial):
    # Hora del resultado
    time = datetime.now()
    print("\nHora Resultado Intermedio:", time.strftime('%d-%m-%H:%M'), time)
    print("Trial,Costo,Parametros")
    for t in study.trials:
        print(f"{t.number},{t.value},{t.params}")


def objective(trial):

    args = obtener_parametros()
    # Sugerir Parámetros 
    args.repeats_temp = trial.suggest_int("repeats_temp", 10,  50)   #de 10 a 50 pasos de 1   #log=True
    args.ini_temp = trial.suggest_int("ini_temp", 1, 50) 
    args.min_temp = trial.suggest_float("min_temp", 0.001, 0.097) 
    args.alpha_func_cool = trial.suggest_float("alpha_func_cool", 0.80, 0.96, step=0.01)  # 0.8 a 0.95
    args.num_vecinos_LS = trial.suggest_int("num_vecinos_LS", 100, 500, log=True)  #número de vecinos
    args.umbral_correlacion = trial.suggest_float("umbral_correlacion", 0.5, 0.9, step=0.1)  # (de 5 % a la mitad)
    args.gene_initial_prob = trial.suggest_float("gene_initial_prob", 0.05, 0.5)  # (de 5 % a la mitad)

    datasets  = ['Lung_GSE63459', 'Prostate_GSE6919_U95C', 'Colorectal_GSE44861']
    positivos  = ['adenocarcinoma','primary_prostate_tumor','tumoral']
    negativos  =['normal','normal','normal']


    # Seleccionar dataset
    ind_data = np.random.randint(len(datasets))
    file_in = './data/data_original/Instances/'+datasets[ind_data]+'.csv'
    
    # Leer dataset seleccionado
    features, dataX, targetY = io.leer_dataset_cumida_2clases(file_in , ",", positivos[ind_data], negativos[ind_data])
    
    # Aplicar Filtros
    args.significancia = 0.05 # nivel de significancia del 5% para seleccionar genes. 
    args.n_jobs_pv_rf=2
    args.max_iterations_rf = 100
    args.cv_rf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5) 
    selected_genes_FILTER, dataX_reducido = filtros.FS_metodo_filter(features, dataX, targetY, args)    
    
    # Ejecutar SA
    args.cv_fitness = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1, 1000000))
    args.max_iterations_fitness = 1000
    args.n_jobs_fitness=5
    selected_features, best_costo_SA = sa.simulated_annealing(selected_genes_FILTER, dataX_reducido, targetY, trial,args) # selected_features son los nombres de los genes
            
    return  best_costo_SA  



#      ----------------------------------------------------------------
#                           EJECUCIÓN DE LA OPTIMIZACIÓN DE LOS PARÁMETROS
#   ------------------------------------------------------------------------

if __name__ == "__main__":

    time =  datetime.now()
    print("\nHora Inicio ejecución:",time.strftime('%d%m%H%M'), time)
    num_cores = multiprocessing.cpu_count()
        
    # Estudio Optuna 
    pruner = SuccessiveHalvingPruner()
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="minimize",  pruner=pruner)  #  o "maximize"   # pruner=MedianPruner(), SuccessiveHalvingPruner(), ThresholdPruner(lower=threshold)
                                                                                                                                                            
    # Paralelización utilizando múltiples procesos
    tiempo_max = 57600 #16 horas
    test_timeout  = 300 # 5 minutos
    study.optimize(objective, callbacks=[logging_callback],  n_trials=300, n_jobs= 30, timeout=tiempo_max ) 

    # Obtener los mejores parámetros encontrados
    print("Mejores parámetros:", study.best_params)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nEstadísticas del study al terminar el proceso: ")
    print("Número de ensayos (trails) terminados: ", len(study.trials))
    print("Número de Ensayos podados: ", len(pruned_trials))
    print("Número de Ensayos completos (sin podar): ", len(complete_trials))
    print("Mejor Ensayo:")
    trial = study.best_trial
    print("  Valor: ", trial.value)
    print("  Parametros: ")
    for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    dir_nom_file_gen = './resultados_SA/Tuning_Optuna/'+time.strftime('%d%m%H%M')
    
    # (1) Traza  el valor de la función objetivo de cada trial a lo largo del tiempo. 
    #   Permite ver cómo los resultados han mejorado a medida que avanzan los trials. 
    #   Los trials están ordenados por el tiempo en que se completaron, y los valores más bajos (si es minimize) indican un mejor rendimiento. 
    #   Una tendencia descendente en un problema de minimización, por ejemplo, indica una mejora en la optimización.
    plot_optimization_history(study).write_html(dir_nom_file_gen +"_optimization_history.html")
    
    # (2):  Se puede ver cómo cambian los valores  intermedios a lo largo de cada trial y cómo evolucionan los algoritmos antes de llegar a su conclusión.
    plot_intermediate_values(study).write_html(dir_nom_file_gen +"_intermediate_values.html")
    
    # (3):  Permite visualizar las relaciones de alto nivel entre los parámetros y los resultados de las ejecuciones (trials).
    #   Cada línea en el gráfico representa una ejecución diferente del algoritmo con un conjunto específico de valores para los parámetros..
    #   Ayuda a visualizar cómo la combinación de diferentes parámetros afecta el rendimiento, lo cual es útil para identificar tendencias y 
    #   dependencias entre parámetros (fijarse en los trials cuyo color es más intenso y la tendencia (colores claros) para los que tienen menores resultados
    plot_parallel_coordinate(study).write_html(dir_nom_file_gen +"_parallel_coordinate.html")
    
    # (4): Cada gráfico en el "slice plot" corresponde a un parámetro y muestra cómo cambia la función objetivo en relación con ese parámetro, 
    #   manteniendo los otros fijos. Esto ayuda a identificar rangos de parámetros que llevan a un mejor rendimiento.
    plot_slice(study).write_html(dir_nom_file_gen+"_slice.html")
    
    # (5): Indica cuán importante es cada parámetro para alcanzar los mejores resultados de la función objetivo. 
    #   Los parámetros con una mayor importancia son aquellos que, cuando ajustados, tienen el mayor impacto en los resultados del modelo. (¿CÓMO LO OBTIENE?)
    plot_param_importances(study).write_html(dir_nom_file_gen +"_param_importances.html")
    
    # (6): Muestra la función de distribución empírica (EDF) de los resultados de la función objetivo.
    #   Permite evaluar la probabilidad de que la función objetivo alcance o exceda ciertos valores, lo cual es útil para entender 
    #   la distribución del rendimiento de los modelos a lo largo del estudio.
    plot_edf(study).write_html(dir_nom_file_gen+"_edf.html")
    
    # (7): Cada eje horizontal representa uno de los parámetros ajustados durante el proceso de optimización.
    # Cada punto representa una configuración particular de los parámetros de un trial. a barra de colores a la derecha muestra la escala, donde los colores más oscuros (azul) representan mejores resultados
    # Para cada parámetro (del eje X) buscar rangos donde se encuentran los puntos azules (buenos)
    # Para cada parámetro (del eje X) buscar rangos donde se encuentran los puntos rojos (malos)
    
    plot_rank(study).write_html(dir_nom_file_gen+"_rank.html")
    
    # (8):  representación visual del tiempo que cada trial ha tomado durante la ejecución de tu estudio de optimización
    plot_timeline(study).write_html(dir_nom_file_gen+"_timeline.html")

    time2 = datetime.now()
    minutos_ejecucion = time2 - time
    print("Tiempo de ejecución:", (minutos_ejecucion))


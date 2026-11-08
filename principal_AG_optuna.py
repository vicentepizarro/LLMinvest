import numpy as np
import time
import optuna
from datetime import datetime
from tools_mh.input_output_data import leer_dataset_cumida_2clases
from FS_optimizador_AG import algoritmo_genetico_optuna  as ag #AG módulo 
from tools_mh  import FS_metodos_filtros as filtros
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import argparse # para obtener parámetros de la linea de comando
from optuna.visualization import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_slice,
    plot_param_importances,
    plot_edf,
    plot_rank,
    plot_timeline
)
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from joblib import Parallel, delayed

def establecer_semilla(semilla):
    np.random.seed(semilla)
    return

def obtener_parametros():
    par = argparse.ArgumentParser()
    par.add_argument('--num_eval_max', help="número máximo de evaluación a la FO", type=int, default=20)
    par.add_argument('--umbral_correlacion', help="umbral de alta correlación para filtrar matriz de correlación", type=float, default=0.70)  #[0.5, 0.9]
    par.add_argument('--num_vecinos_LS', help="Número de veces que se ejecuta LS", type=int, default=100)   #[100-500]
    par.add_argument('--gene_initial_prob', help="probabilidad de que el gen sea seleccionado inicialmente", type=float, default=0.02)  #[0.05,0.5]
    par.add_argument('--pop_size', help="tamaño de la pobla", type=int, default=10)     #[ 10, 100]
    #-
    par.add_argument('--percentage_select', help="porcentaje  para la selección de los padres", type=float, default=0.1) #[0.10, 0.50]
    par.add_argument('--crossover_rate', help="probabilidad de cruce", type=float, default=0.1)     #[0.1, 0.9]
    par.add_argument('--mutation_rate', help="Probabilidad de mutación de cada individuo de la población", type=float, default=0.20)   #[0.1, 0.30]
    par.add_argument('--gene_mutation_rate', help="Probabilidad de mutación de cada gen de un individuo", type=float, default=0.05)   #[ 0.01, 0.05]
    par.add_argument('--isLS', help="Si ejecuta búsqueda loca o no", type=str, default='si')    
    return par.parse_args()
    
# Función objetivo para Optuna
def objective(trial):
    args = obtener_parametros()
    #args.num_eval_max   ===> parametro de entrada en línea de comando
    #args.umbral_correlacion = trial.suggest_float('umbral_correlacion', 0.5, 0.9, step=0.1) # ES FIJO
    args.num_vecinos_LS = trial.suggest_int('num_vecinos_LS', 100, 500, log=True)  # 100-500
    args.gene_initial_prob  = trial.suggest_float('gene_initial_prob', 0.05, 0.5)
    args.pop_size = trial.suggest_int('pop_size', 10, 100)  #10, 100
    args.percentage_select = trial.suggest_float('percentage_select', 0.10, 0.50)   #para operador de selection 
    args.crossover_rate = trial.suggest_float('crossover_rate', 0.1, 0.9) # probabilidad de que un gen (posición en el cromosoma) sea tomado del primer padre en lugar del segundo.
    args.mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.30)  # % de individuos que mutan
    args.gene_mutate_rate = trial.suggest_float('gene_mutate_rate', 0.01, 0.05)  # % de genes que mutan de cada lindividuo
    
    datasets  = ['Lung_GSE63459', 'Prostate_GSE6919_U95C', 'Colorectal_GSE44861']
    positivos  = ['adenocarcinoma','primary_prostate_tumor','tumoral']
    negativos  =['normal','normal','normal']

    # SELECCIONAR DATASET de forma aleatoria
    ind_data = np.random.randint(len(datasets))
    file_in = './data/data_original/Instances/'+datasets[ind_data]+'.csv'
    args.nom_dataset = datasets[ind_data]
    
    # CARGAR DATASET 
    features, dataX, targetY = leer_dataset_cumida_2clases(file_in , ",", positivos[ind_data], negativos[ind_data])
    #print("nom_dataset:",datasets[ind_data] ,"Tumor:", np.sum(targetY), "Normal:", np.sum(targetY==False))

    # APLICAR FILTROS. Estos filtros varian significativamente al cambiar los parámetros de cv_rf y de la significancia estadística
    '''
    confianza_estadistica = 0.95 ó 0.99 ó 0.999
    Seleccionar entre los siguientes niveles de significancia estadística para seleccionar genes
        - 0.05 (5%)    :  puede resultar en un número relativamente alto de falsos positivos, especialmente cuando se analizan un gran número de genes.
        - 0.01 (1%)     :  control más estricto de falsos positivos.
        - 0.001 (0.1%) : estudios con gran cantidad de datos, como en microarrays con decenas de miles de genes, 
                                para reducir aún más la posibilidad de falsos positivos. 
                                Este nivel es más conservador y es útil cuando se requiere una gran confianza en los genes seleccionados.
    '''
    args.significancia = 0.05 # nivel de significancia del 5% para seleccionar genes. 
    args.n_jobs_pv_rf=2
    args.max_iterations_rf = 100
    args.cv_rf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5) 
    #selected_genes_FILTER, dataX_reducido = filtros.FS_metodo_filter(features, dataX, targetY, args)    

    # EJECUTAR OPTIMIZADOR
    args.semilla = np.random.randint(1, 1000000)
    establecer_semilla(args.semilla)
    args.cv_fitness = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.semilla)
    args.max_iterations_fitness = 1000
    args.n_jobs_fitness=5
    #fitness_best_AG   = ag.AG(dataX_reducido, targetY, trial, args)    
    fitness_best_AG   = ag.AG(dataX, targetY, trial, args) # Sin filtros  
    
    return fitness_best_AG 

#
# OPTIMIZAR CON TRABAJOS PARALELOS 
def run_trial(study, n_trials):
    for _ in range(n_trials):
        try:
            study.optimize(objective, n_trials=1)
        except Exception as e:
            print(f"Error during optimization: {e}")
    return study

if __name__ == "__main__":

    start_time = time.time()
    pruner = SuccessiveHalvingPruner()
    #study = optuna.create_study(direction='maximize', pruner=pruner)
    #.- Usar storage para asegurarse que los triales se guarden en un único lugar y evitar problemas de concurrencia
    study = optuna.create_study(direction='maximize', pruner=pruner, storage="sqlite:///tunning_parametros.db", study_name="AG_study", load_if_exists=True)

    
    n_trials = 100
    n_jobs = 10  #Bebe ser divisible por n_trials (en este caso 100/10 = 10  la división es exacta)
    #.- Ejecutar 
    def run_study_trials():
        return run_trial(study, n_trials // n_jobs)

    results = Parallel(n_jobs=n_jobs)(delayed(run_study_trials)() for _ in range(n_jobs) )
    
    dir_nom_file_gen = './resultados_AG/Tuning_Optuna/'+time.strftime('%d%m%H%M')
    plot_optimization_history(study).write_html(dir_nom_file_gen +"_optimization_history.html")
    plot_intermediate_values(study).write_html(dir_nom_file_gen +"_intermediate_values.html")
    plot_parallel_coordinate(study).write_html(dir_nom_file_gen +"_parallel_coordinate.html")
    plot_slice(study).write_html(dir_nom_file_gen+"_slice.html")
    plot_param_importances(study).write_html(dir_nom_file_gen +"_param_importances.html")
    plot_edf(study).write_html(dir_nom_file_gen+"_edf.html")
    plot_rank(study).write_html(dir_nom_file_gen+"_rank.html")
    plot_timeline(study).write_html(dir_nom_file_gen+"_timeline.html")

    time2 = time.time()
    minutos_ejecucion = (time2 - start_time)/60
    print("Tiempo de ejecución:", minutos_ejecucion, "minutos")




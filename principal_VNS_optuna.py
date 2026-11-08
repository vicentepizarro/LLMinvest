# principal_VNS_optuna.py
# -------------------
import optuna
import numpy as np
import logging
from datetime import datetime
from FS_optimizador_VNS import config_param_vns as param
from tools_mh import input_output_data as io
from Filtros import FS_metodos_filtros as FS_filtros
from FS_optimizador_VNS import variable_neighborhood_search_optuna as vns
from Operadores import Operadores_solUnica as op
import time
import sys
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, plot_slice, plot_param_importances, plot_edf, plot_rank, plot_timeline
from optuna.trial import TrialState
from optuna.samplers import TPESampler, RandomSampler, GridSampler  # ver: https://datagy.io/python-optuna/
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

def logging_callback(study, trial):
    # Hora del resultado
    time =  datetime.now()
    print("\nHora Resultado Intermedio:",time.strftime('%d-%m-%H:%M'), time)
    print("Trial,Costo,Parametros")
    for t in study.trials:
        print(f"{t.number}, {t.value}, {t.params}")

def objective(trial):

        args = param.obtener_parametros()

        #Dataset para el trial:  Ejemplo: --nom_dataset= Colorectal_GSE44861,Colorectal_GSE448612Colorectal_GSE44863
        datasets  = args.nom_dataset.split(",")
        positivos  = args.nom_positivo.split(",")
        negativos  = args.nom_negativo.split(",")        
        
        # Seleccionar dataset
        ind_data = np.random.randint(len(datasets))
        file_in = './data/data_original/Instances/'+datasets[ind_data]+'.csv'

        # Sugerir Parámetros
        args.num_veces_LS = trial.suggest_int("num_veces_LS", 100, 500, log=True)  # sin LS y con LS va igual
        args.umbral_correlacion = trial.suggest_float("umbral_correlacion", 0.5, 0.9, step=0.1)  
        args.prob_gen_true = trial.suggest_float("prob_gen_true", 0.05, 0.5)  # (de 5 % a la mitad)

        # Leer data set seleccionado
        features, dataX, targetY = io.leer_dataset_cumida_2clases(file_in , args.sep, positivos[ind_data], negativos[ind_data])

        # Aplicar Filtros
        selected_genes_FILTER, dataX_reducido = FS_filtros.FS_metodo_filter(features, dataX, targetY)    
    
        # Ejecutar VNS
        vns.set_args(args) #habilito los parámetros al módulo 
        op.set_args(args) #habilito los parámetros al módulo 
        selected_features_VNS, best_costo = vns.variable_neighborhood_search(selected_genes_FILTER, dataX_reducido, targetY, trial) #args

        return  best_costo  #retornar el mean_cost si se requiere



if __name__ == "__main__":

        time =  datetime.now()
        print("\nHora Inicio ejecución:",time.strftime('%d-%m-%H%M'), time)
            
        # Estudio Optuna 
        pruner = SuccessiveHalvingPruner()
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction="minimize",  pruner=pruner)  #  o "maximize"   
                                                                                                                                                                
        # Paralelización utilizando múltiples procesos
        tiempo_max = 57600 # horas
        test_timeout  = 300 # 5 minutos
        study.optimize(objective, callbacks=[logging_callback],  n_trials=300, n_jobs= 100, timeout=tiempo_max )  #n_jobs= 50, 

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

        dir_nom_file_gen = "./resultados_VNS/Tuning_Optuna/" + time.strftime('%d%m%H%M')
        
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




import numpy as np
import os
import time
from datetime import datetime
import argparse 
from sklearn.model_selection import RepeatedStratifiedKFold
import copy
import EVALfitness.eval as eval #referencia para usar cualquier cosa de eval.py

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback

from optuna.pruners  import MedianPruner  #ó
from optuna.pruners import PercentilePruner

from MOoptimization.NSGAII import NSGA2
from MOoptimization.MOEAD import MOEAD
from MKNN.MKNN import grafo_guiador_LS

parametrizar = True

def leer_dataFilter_y_matrizSim(file_filter, args): # usado para optmizar (una sola matriz y un grafo cuando corresponda)
    sims = {}
    print(f"Cargando {file_filter} ......")
    with np.load(file_filter, allow_pickle=False) as data:
        X_filt   = data["X"]  # NO ESTÁ ESCALADA
        genes_filt = data["genes"].ravel()
        targetY = data["Y"].ravel()

        if args.metodo in ['MOEAD+mKNNG-LS', 'NSGA-II+mKNNG-LS']:
            key_npz = f"S_{args.similitud}"   # "S_pearson" o "S_spearman"
            sims[args.similitud] = data[key_npz]
            print(f'Matriz {args.similitud} OBTENIDA DE DISCO: {data[key_npz].shape}')
        print("Dimensiones Dataset filtrado (Sin escalar) OBTENIDAS DE DISCO:", X_filt.shape, genes_filt.shape)
    return X_filt, genes_filt, targetY, sims

if __name__ == "__main__":

    if parametrizar:
        def obtener_parametros_base():
            par = argparse.ArgumentParser()
            par.add_argument('--semilla', help="Semilla para reproducir ejecución", type=int)
            par.add_argument('--num_evals_max', help="Máximo número de evaluaciones objetivo", type=int)
            par.add_argument('--cv_folds', type=int, help="Número de folds CV (estratificado)") 
            par.add_argument('--epsilon', type=float, help="ε para aceptar pequeños empeoramientos")  
            par.add_argument('--metodo', help="Algoritmo con NSGA2 o NSGA-II+mKNNG-LS", type=str)
            par.add_argument('--clasificador', type=str, choices=["NB", "RF", "DT", "RL"], required=True)
            par.add_argument('--isOptuna', help="Algoritmo con optuna o sin LS", type=str, default='si')
            par.add_argument('--paralelo', help="es paralelo o no", type=str, default='no')
            par.add_argument('--n_jobs_fitness', type=int, help="Número de de hilos para el fitness")
            par.add_argument('--n_jobs_ls', type=int, help="Hilos para la búsqueda local (threading)")
            par.add_argument('--version', help="Identifivador de la ejecución paralela", type=int, default=1) 
            par.add_argument('--moead_batch_size', type=int, required=True, help="Tamaño del lote de subproblemas en MOEA/D")
            par.add_argument('--optuna_storage_url', help="url base de datos optuna", type=str)
            par.add_argument('--study_name', help="Nombre del estudio donde están los parámetros sugeridos y la mejor alternativa", type=str)
            return par.parse_args()



        def precargar_datasets(args_base):
            datasets  = ['Lung_GSE63459', 'Prostate_GSE6919_U95C'] #, 'Liver_GSE14520_U133A'
            positivos = ['adenocarcinoma','primary_prostate_tumor'] #,'HCC'
            negativos = ['normal','normal','normal']
            k_optimos = [33,33] #,35
            similitd_s = ['pearson','pearson'] #,'pearson'
            datasets_param = []
            for ind, ds in enumerate(datasets):
                args_ds = copy.deepcopy(args_base)
                args_ds.nom_dataset = ds
                args_ds.pos = positivos[ind]
                args_ds.neg = negativos[ind]
                args_ds.k_optimo = k_optimos[ind]
                args_ds.similitud = similitd_s[ind]
                file_filter = './_DATA/FILTERED/PARAM/'+args_ds.nom_dataset+'_filter.npz'
                file_grafo_disco = f"./_DATA/GRAFOS/PARAM/{args_ds.nom_dataset}_{args_ds.similitud}_{args_ds.k_optimo}-MNN_grafo.gpickle"

                G, comunidad_of_nodos, inds_comunidades, top_nodes_por_comunidad = None, None,None,None
                dataX_reducido_NO_scaled, features_FILTER, targetY, sims = leer_dataFilter_y_matrizSim(file_filter, args_ds)

                if args_ds.metodo in ["NSGA-II+mKNNG-LS", "MOEAD+mKNNG-LS"]:
                    sim_mat = sims.get(args_ds.similitud)
                    G, comunidad_of_nodos, inds_comunidades, top_nodes_por_comunidad= grafo_guiador_LS(args_ds.semilla, sim_mat,args_ds.nom_dataset, args_ds.k_optimo, features_FILTER, file_grafo_disco)

                datasets_param.append({
                    "nom_dataset": ds,
                    "pos": args_ds.pos,
                    "neg": args_ds.neg,
                    "dataX_reducido_NO_scaled": dataX_reducido_NO_scaled,
                    "features_FILTER": features_FILTER,
                    "targetY": targetY,
                    "comunidad_of_nodos": comunidad_of_nodos,
                    "top_nodes_por_comunidad": top_nodes_por_comunidad,
                    "inds_comunidades": inds_comunidades,
                    "G_mutuo": G
                })
            return datasets_param
        
        #. ____________________________________________________________________
        def objective(trial):
            #.- 1) Construir args_trial limpio por trial (no reutilizar global)   
            global args_base, param_ds
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)   # workers frescos por trial → ppto justo
            t0 = time.perf_counter()
            args_trial = copy.deepcopy(args_base) 
            # .--------------------------------------
            # PARÁMETROS EN LOS 4 MÉTODOS: "NSGA-II", "NSGA-II+mKNNG-LS", "MOEAD", "MOEAD+mKNNG-LS"
            # Para MOEA/D cada individuo es la mejor solución encontrada para un subproblema
            # Para MOEA/D  T es el tamaño del vecindarios: cuántos subproblemas (individuos) vecinos tiene el subproblema i definidos en B(i)
            # Los vecinos B(i) se usan para:
            #   - (a) elegir los dos padres 
            #   - (b)  (b) recibir al hijo en la actualización.
            #.--------------------
            args_trial.pop_size            = trial.suggest_int('pop_size',  50, 200)  # es Lo mismo que N subproblemas en MOEAD
            args_trial.crossover_rate      = trial.suggest_float('crossover_rate', 0.7, 1.0, step=0.01)  
            args_trial.gene_mutation_rate  = trial.suggest_float('gene_mutation_rate', 0.001, 0.05, step=0.001)
            args_trial.gene_initial_prob   = trial.suggest_float('gene_initial_prob', 0.3, 0.5, step=0.01)
            args_trial.LS_frac_children = 0
            if args_trial.metodo in ["NSGA-II+mKNNG-LS", "MOEAD+mKNNG-LS"]: 
                # En nsga2 existen N hijos por generación y LS_frac_children es la fracción de este lote que recibe LS
                # En MOEA/D LS recibe esa misma fracción pero asociada a los subproblemaas
                args_trial.LS_frac_children = trial.suggest_float('LS_frac_children', 0.05, 0.40, step=0.01)  

            args_trial.moead_T = 0
            if args_trial.metodo in ["MOEAD", "MOEAD+mKNNG-LS"]:
                # T absoluto: Zhang&Li 2007 usan T=20; bueno en [10,50], malo si es muy
                # chico (poca exploración) o grande en combinatorios. Sin nr acotamos a [10,25].
                args_trial.moead_T = trial.suggest_int('moead_T', 10, 25)
            args_trial.mutation_rate = 1
            if args_trial.metodo in ["NSGA-II", "NSGA-II+mKNNG-LS"]:            
                args_trial.mutation_rate= trial.suggest_float('mutation_rate', 0.01, 0.20, step=0.01)

            # 2 datasets por trial, de menor a mayor tamaño. El pruning corta un trial malo
            # ya tras el 1er dataset (Prostate_U95C), sin gastar el budget del 2do (Lung).
            sizes = [it["dataX_reducido_NO_scaled"].size for it in param_ds]
            order = np.argsort(sizes)  # ascendente
            ds_order_names = [param_ds[i]["nom_dataset"] for i in order.tolist()]            
            #.-
            hv_vals = []
            hv_por_ds = {}
            for j, ds_idx in enumerate(order):
                #.-
                eval.clear_ls_caches()
                item = param_ds[ds_idx]
                args_trial_ds = copy.deepcopy(args_trial)
                args_trial_ds.nom_dataset = item["nom_dataset"]
                args_trial_ds.pos = item["pos"]
                args_trial_ds.neg = item["neg"]
                master_ds = int(args_trial_ds.semilla) + 10_000 * int(args_trial_ds.version) + 100 * int(ds_idx)
                seed_cv     = master_ds + 1 # se una en el splits
                rkf = RepeatedStratifiedKFold(n_splits=args_trial_ds.cv_folds, n_repeats=1, random_state=seed_cv)
                splits = list(rkf.split(item["dataX_reducido_NO_scaled"],item["targetY"]))
                #.-
                seed_ea_ds  = master_ds + 2 # se usa en el generador random global de ea
                seed_init_ds= master_ds + 3 # se usa en el generador random problaciones iniciales
                rng_global = np.random.default_rng(seed_ea_ds)
                rng_init   = np.random.default_rng(seed_init_ds)
                #.-
                seed_ls     = master_ds + 4 # Semilla usada en la LS
                seed_clf     = master_ds + 5 # # Semilla del clasificador (RF/DT/ET); también entra al ctx_key
                args_trial_ds.step_offset = j * args_trial_ds.num_evals_max # step en optuna usamos 500000*0 + num_eval para el 1er dataset
                                                                            #                     y 500000*1 + num_eval para el 2do dataset
                args_trial_ds.completed_hv_sum = float(sum(hv_vals))  # para el cálculo de running_mean 
                args_trial_ds.completed_count  = int(len(hv_vals))    # para el cálculo de running_mean
                print(f"[trial={trial.number:03d} ds={ds_idx}] "
                      f"dataset={args_trial_ds.nom_dataset} "
                      f"step_offset={args_trial_ds.step_offset}",
                      flush=True)

                if args_trial.metodo in ["NSGA-II", "NSGA-II+mKNNG-LS"]: 
                    _, _, Hv = NSGA2(item["G_mutuo"], item["dataX_reducido_NO_scaled"], item["targetY"], item["comunidad_of_nodos"], item["inds_comunidades"], 
                                    item["top_nodes_por_comunidad"], trial=trial, args=args_trial_ds, splits=splits, seed_clf=seed_clf, 
                                    rng_global=rng_global, rng_init=rng_init, seed_ls=seed_ls)
                    
                if args_trial.metodo in ['MOEAD', 'MOEAD+mKNNG-LS']:
                    _, _, Hv = MOEAD(item["G_mutuo"], item["dataX_reducido_NO_scaled"], item["targetY"], item["comunidad_of_nodos"], item["inds_comunidades"], 
                                    item["top_nodes_por_comunidad"], trial=trial, args=args_trial_ds, splits=splits, seed_clf=seed_clf, 
                                    rng_global=rng_global, rng_init=rng_init, seed_ls=seed_ls)
                    
                print(f"[trial={trial.number:03d} ds={ds_idx}] HV_final={Hv:.6f}", flush=True)
                hv_vals.append(Hv)
                hv_por_ds[item["nom_dataset"]] = float(Hv)

            #Trazabilidad por trial
            time_min = float((time.perf_counter() - t0) / 60.0)
            mean_hv = float(np.mean(hv_vals))
            trial.set_user_attr("hv_por_dataset", hv_por_ds)   # dict {ds: hv}
            trial.set_user_attr("trial_time_min", time_min)

            print(f"[trial={trial.number:03d}]  hv_por_ds={hv_por_ds}  meanHV={mean_hv:.6f}  time_min={time_min:.2f}", flush=True)
            return mean_hv

        def objective_r1(trial):
            global args_base, param_ds
            args_base = args_r1                      # <- objective leerá esta base
            if 'param_ds' not in globals():          # precargar sólo una vez los datasets
                param_ds = precargar_datasets(args_base)
            return objective(trial) # Ejecuta  objective original



        #.-________________________________________________________________________________________________
        #
        #               EJECUTAR ESTUDIO:   
        #                             
        #._________________________________________________________________________________________________
        #

        args_r1 = obtener_parametros_base()
        args_r1.n_trials = 30  # número de ensayos optuna
        args_r1.n_startup_trials=8 #controla CUÁNTOS TRIALS completos deben existir antes de permitir pruning.
        args_r1.percentile=40.0 # # umbral (obtenido de la lista de running_mean de los 7 trails anteriores) para decidir si se poda el trail actual según el running_mean reportado.
        args_r1.n_warmup_steps = int(args_r1.num_evals_max * 0.15) # controla CUÁNTO debe avanzar CADA TRIAL antes de permitir pruning (por dataset)
        args_r1.interval_steps = int(args_r1.num_evals_max * 0.05) # frecuencia de revisión del pruning 


        # Corta un trial antes de terminar usando la mediana como referencia (poda rápido; simple; ahorra tiempo)
        # usa siempre el percentil 50, es decir, la mediana.
        pruner1 = MedianPruner(
                n_startup_trials=args_r1.n_startup_trials, #controla CUÁNTOS TRIALS completos deben existir antes de permitir pruning.
                n_warmup_steps=args_r1.n_warmup_steps, # controla CUÁNTO debe avanzar CADA TRIAL antes de permitir pruning
                interval_steps=args_r1.interval_steps # frecuencia de revisión del pruning
                )
        # Corta un trial si está dentro del percentile peor (que he definido ) de los trials comparables en ese step
        # percentile usa el running_mean reportado:
        # running_mean = (args.completed_hv_sum + hv0) / (args.completed_count + 1) = promedio acumulado del HV considerando los datasets ya evaluados
        pruner2 = PercentilePruner(percentile=args_r1.percentile, # umbral (obtenido de la lista de running_mean de los 7 trails anteriores) para decidir si se poda el trail actual según el running_mean reportado.
                                   n_startup_trials=args_r1.n_startup_trials, #TRIALS completos deben existir antes de permitir pruning
                                   n_warmup_steps=args_r1.n_warmup_steps, # para cada trial, no podar antes del paso 75000 (0.15*500000= 75000)
                                   interval_steps=args_r1.interval_steps # revisa cada 5% (cada 25000 evaluaciones)
                                   ) 
        """ En este caso los steps optuna revisados son: 75000, 100000, 125000, 150000, ...
        # Ejemplo step 100000, Optuna mira qué running_mean tuvieron los trials anteriores en ese mismo step.
          Supongamos estos valores:
            Trial 0, step 100000 → running_mean = 0.41
            Trial 1, step 100000 → running_mean = 0.48
            Trial 2, step 100000 → running_mean = 0.52
            Trial 3, step 100000 → running_mean = 0.55
            Trial 4, step 100000 → running_mean = 0.60
            Trial 5, step 100000 → running_mean = 0.63
            Trial 6, step 100000 → running_mean = 0.69
            Trial 7, step 100000 → running_mean = 0.74        
            Ordenados: 0.41, 0.48, 0.52, 0.55, 0.60, 0.63, 0.69, 0.74
            => el percentile 40.0 para estos 8 valores está aprox en la posición (desde 0): (8-1)*0.40=2.8 ==> aprox en 3 => 0.55
            El PercentilePruner poda si el mejor valor intermedio del trial actual cae en la parte baja según ese percentil, comparando contra trials previos en el mismo step.
            Si se reporta un running_mean = 0.50 se poda el trial 8
        """
        # Crear study si no existe
        # Usamos: TPESampler (a diferencia de RandomSampler) ya que no prueba valores al azar. Primero observa qué valores dieron buenos resultados y luego propone 
        #         nuevos valores más parecidos a los buenos que a los malos. Ajusta modelos probabilísticos para separar configuraciones buenas y malas
        sampler_seed = args_r1.semilla + os.getpid()
        study_r1 = optuna.create_study(direction="maximize", 
                                    study_name=args_r1.study_name, 
                                    storage=args_r1.optuna_storage_url,
                                    sampler=TPESampler(seed=sampler_seed, constant_liar=True), 
                                    pruner=pruner1, 
                                    load_if_exists=True
                                    )
        #trazabilidad por estudio
        study_r1.set_user_attr("n_trials", int(args_r1.n_trials))
        study_r1.set_user_attr("n_startup_trials", int(args_r1.n_startup_trials))
        study_r1.set_user_attr("percentile", float(args_r1.percentile))
        study_r1.set_user_attr("n_warmup_steps", int(args_r1.n_warmup_steps))
        study_r1.set_user_attr("interval_steps", int(args_r1.interval_steps))
        study_r1.set_user_attr("metodo", args_r1.metodo)
        study_r1.set_user_attr("clasificador", args_r1.clasificador)
        study_r1.set_user_attr("num_evals_max", int(args_r1.num_evals_max))
        study_r1.set_user_attr("cv_folds", int(args_r1.cv_folds))
        study_r1.set_user_attr("epsilon", float(args_r1.epsilon))
        study_r1.set_user_attr("n_jobs_fitness", int(args_r1.n_jobs_fitness))
        study_r1.set_user_attr("n_jobs_ls", int(args_r1.n_jobs_ls))
        study_r1.set_user_attr("semilla", int(args_r1.semilla))
        #.-

        study_t0 = time.perf_counter()
        print(">>> Optimizando hiperparámetros...", study_t0, flush=True)

        max_trials_cb = MaxTrialsCallback(args_r1.n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))
        study_r1.optimize(objective_r1, n_trials=None, n_jobs=1, callbacks=[max_trials_cb])

        total_time_trials_faltan_min = float((time.perf_counter() - study_t0) / 60.0)
        study_r1.set_user_attr("total_time_trials_faltan_min", total_time_trials_faltan_min)

        complete_trials = study_r1.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        print(f" Worker terminó. Tiempo local: {total_time_trials_faltan_min} minutos", flush=True)



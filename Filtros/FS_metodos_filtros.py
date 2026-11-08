from sklearn.feature_selection import VarianceThreshold
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed # facilita la paralelización de tareas

#  ---------------------------------------------------------------------------------------------
#           FILTROS
# -------------------------------------------------------------------------
# La varianza es una medida que no depende de la escala de los datos
def eliminar_features_const_o_quasi(X_train, features,umbral): #umbral=0.01
    features_const_model = VarianceThreshold(threshold=umbral)
    features_const_model.fit(X_train)
    #
    # Devuelve el dataset con las columnas eliminadas
    X_train  = features_const_model.transform(X_train)
    # True las conservadas, False las eliminadas
    features_drop = features[~features_const_model.get_support()] 
    features = features[features_const_model.get_support()]
    return X_train, np.where(features_const_model.get_support())[0]
# ---------------------------------------------------------------------------------

def signaficance_filter(data, features_inds, pvalues, significancia):
    #ordenar genes de selected_inds_var según orden de los p-values 
    sorted_indices = np.argsort(pvalues) 
    sorted_pvalues = pvalues[sorted_indices]
    sorted_features_ind = features_inds[sorted_indices] 
    # Genes ordenados por p-value y con p-values < significancia estadística definida
    selected_inds_genes = sorted_features_ind[sorted_pvalues < significancia]  # <alpha
    dataX_reducido = data[:, sorted_pvalues < significancia]
    return dataX_reducido, selected_inds_genes 

# Los árboles de decisión y, por extensión, los bosques aleatorios (Random Forest), 
# no son sensibles a la escala de los datos. Esto se debe a que estos algoritmos dividen 
# los datos según umbrales de características individuales, y la comparación de valores es independiente de su escala.
def importancia_filter(data, selected_inds, target, args): #args.max_iterations_rf
    rf = RandomForestClassifier(n_estimators=args.max_iterations_rf, random_state=args.semilla)
    rf.fit(data, target)
    importances = rf.feature_importances_
    important_indices = np.where(importances > 0)[0] 
    selected_inds_importancia = selected_inds[important_indices]
    dataX_reducido_importancia = data[:, important_indices]
    return dataX_reducido_importancia, selected_inds_importancia
    
def importancia_filter_repeat(data, selected_inds, target, args): #args.max_iterations_rf, args.cv_rf 
    importancias_acumuladas = np.zeros(data.shape[1])  #folds ejecutados =  cv_rf.get_n_splits()
    rf = RandomForestClassifier(n_estimators=args.max_iterations_rf, random_state=args.semilla) # se crea una instancia del modelo RF
    for ind_train, _ in args.cv_rf.split(data, target): # Forma de CV para quedarme con el modelo
        rf.fit(data[ind_train], target[ind_train])
        importancias_acumuladas += rf.feature_importances_ #acumula la importancia
    #importancias_promedio = importancias_acumuladas / (cv_rf.get_n_splits()) # no es necesario
    # conservo genes con importancia acumulada distinto de cero
    indices_con_importancias_mayor_cero = np.where(importancias_acumuladas > 0)[0] 
    # índices de selected_inds_sig que están en indices_con_importancias_mayor_cero
    selected_inds_importancia = selected_inds[indices_con_importancias_mayor_cero]
    dataX_reducido_importancia = data[:, indices_con_importancias_mayor_cero]
    return dataX_reducido_importancia, selected_inds_importancia
    

# Determinar si hay diferencias estadísticamente significativas en la distribución de los valores de expresión entre los grupos
# Este test asume que las observaciones son independientes y que las distribuciones de los dos grupos son comparables.
# ¿Considera aplicar una corrección por múltiples comparaciones (como Bonferroni o FDR) si estás evaluando muchos genes, para controlar la tasa de falsos positivos.
# # No es necesario escalar los valores para este test estadístuco
# El test de Mann-Whitney es una prueba estadística no paramétrica que compara distribuciones y es robusta a la escala de los datos
# No es necesario escalar
def calcular_pvalues(sanos, enfermos): # test estadístico Mann-Whitney U (también conocido como Wilcoxon rank-sum test)
    pvalues = mannwhitneyu(sanos, enfermos).pvalue # recibe sanos y enfermos separados para cada gen
    return pvalues

#
#           PRINCIPAL
#
def FS_metodo_filter(features, dataX, targetY, args):

    # Filtro basado en varianza
    dataX_reducido, selected_inds_genes = eliminar_features_const_o_quasi(dataX, features, args.umbral_varianza)

    #
    # Filtro basado  en significancia estadística con ajuste de los p-values 
    #
    geneSanos = dataX_reducido[targetY == 0, :]
    geneEnfermos = dataX_reducido[targetY == 1, :]
    pvalues = np.array(Parallel(n_jobs=args.n_jobs_pv_rf)(delayed(calcular_pvalues)(geneSanos[:,ind_gen], geneEnfermos[:, ind_gen]) for ind_gen in range(dataX_reducido.shape[1])) )
    mask_pvalues_sig, pvalues_FDR, _, _ = multipletests(pvalues, alpha=args.significancia, method='fdr_bh')
    selected_inds_sig_FDR = selected_inds_genes[mask_pvalues_sig]
    dataX_reducido_sig_FDR = dataX_reducido[:, mask_pvalues_sig]
    dataX_reducido_sig, selected_inds_genes_sig = signaficance_filter(dataX_reducido, selected_inds_genes, pvalues, args.significancia)  #np.array(pvalues)

    #
    # Filtro basado en Importancia Randon Forest
    #
    dataX_reducido_rf, selected_inds_genes_rf  = importancia_filter_repeat(dataX_reducido_sig, selected_inds_genes_sig, targetY, args)

    return selected_inds_genes_rf, dataX_reducido_rf 
    
    

    

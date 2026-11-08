# -------------------
# config_param.py
# --------------------
#import time
#from datetime import datetime
import argparse
def obtener_parametros():
    par = argparse.ArgumentParser()
    par.add_argument('--nom_dataset', help="dataset de entrada", type=str, default='Colorectal_GSE25070')   
    par.add_argument('--nom_positivo', help="nombre target muestra positiva", type=str, default='tumoral')
    par.add_argument('--nom_negativo', help="nombre target muestra negativa", type=str, default='normal')
    par.add_argument('--is_data_filter', help="¿se han aplicado métodos de FS Filter?", type=str, default='no')
    par.add_argument('--is_plot', help="¿mostrar plot de resultados?", type=str, default='no') #termina en ${12} desde instances
    par.add_argument('--sep', help="separador entre datos", type=str, default=',') # \t
    par.add_argument('--num_eval_max', help="número máximo de evaluación a la FO", type=int, default=100000)
    par.add_argument('--umbral_correlacion', help="umbral de alta correlación para filtrar matriz de correlación", type=float, default=0.70)  
    par.add_argument('--repeats_temp', help="número de repeticiones por temperatura", type=int, default=300)
    par.add_argument('--ini_temp', help="temperatura inicial", type=float, default=1000)
    par.add_argument('--min_temp', help="temperatura mínima", type=float, default=0.001)
    par.add_argument('--alpha_func_cool', help="factor (alfa) de función de enfriamiento geometrico", type=float, default=0.99)
    par.add_argument('--num_vecinos_LS', help="Número de veces que se ejecuta LS", type=int, default=100)
    par.add_argument('--prob_gen_true', help="probabilidad de que el gen sea seleccionado inicialmente", type=float, default=0.02)    
    return par.parse_args()



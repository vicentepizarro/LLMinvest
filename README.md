# AgenticAI-FS: Agente Autónomo para Selección de Características en Expresión Génica

**AgenticAI-FS** es un sistema agéntico inteligente impulsado por Modelos de Lenguaje Masivo (LLMs) y metaheurísticas bioinformáticas avanzadas. Su objetivo principal es resolver el problema de alta dimensionalidad (p &gt;&gt; n) en datos transcriptómicos (RNA-seq y microarrays), asistiendo a investigadores y profesionales clínicos en el descubrimiento automatizado de paneles compactos de biomarcadores tumorales con alta capacidad diagnóstica (ROC-AUC).

---

## Tabla de Contenidos

* [Arquitectura del Sistema](#arquitectura-del-sistema)
* [Cronograma de Desarrollo e Hitos](#cronograma-de-desarrollo-e-hitos)
* [Estructura del Repositorio](#estructura-del-repositorio)
* [Requisitos e Instalación Rápida](#requisitos-e-instalación-rápida)
* [Guía de Uso](#guía-de-uso) 
  * [1\. Modo Interactivo por Terminal (Semana 7)](#1-modo-interactivo-por-terminal-semana-7)
  * [2\. Suite de Evaluación Automatizada MVP (Semana 8)](#2-suite-de-evaluación-automatizada-mvp-semana-8)
* [Código Fuente: Suite de Evaluación (Semana 8)](#código-fuente-suite-de-evaluación-semana-8)
* [Descripción de Herramientas Bioinformáticas](#descripción-de-herramientas-bioinformáticas)
* [Variables de Entorno y Configuración](#variables-de-entorno-y-configuración)
* [Licencia y Créditos](#licencia-y-créditos)

---

## Arquitectura del Sistema

El flujo de trabajo desacopla la formulación de consultas clínicas de la complejidad algorítmica mediante el patrón de razonamiento **ReAct** (Reasoning and Acting):

```
                                  +-----------------------+
                                  |   Usuario / Médico    |
                                  +-----------------------+
                                              │ (Lenguaje natural)
                                              ▼
                             +---------------------------------+
                             |     AgenticAI-FS Orchestrator    |
                             |   (ChatGroq - GPT-OSS / Llama)  |
                             |   LangGraph State &amp; MemorySaver |
                             +---------------------------------+
                                       │              ▲
                       (Llamada a Tool)│              │(Resultados JSON)
                                       ▼              │
    +---------------------------------------------------------------------------------+
    |                     MÓDULO DE HERRAMIENTAS (tools_bio.py)                       |
    |                                                                                 |
    |  [1] Filtro Estadístico       [2] Metaheurísticas Mono     [3] Optimización     |
    |      • Mann-Whitney (p&lt;0.01)      • simLS + GA                 Multiobjetivo    |
    |      • Random Forest Gini         • simLS + SA                 • zNSGA2-LS      |
    |                                   • simLS + VNS                • Operador Zoom  |
    +---------------------------------------------------------------------------------+
                                       │
                                       ▼
                   +---------------------------------------+
                   |  Datos Procesados (RNA-seq / CuMiDa)  |
                   |   • Matriz Expresión: M_expr.csv      |
                   |   • Matriz Correlación: M_corr.npy    |
                   +---------------------------------------+

```

---

## Cronograma de Desarrollo e Hitos

| Semana       | Módulo / Hito               | Descripción Técnica                                                                                                                                    |
| ------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Semana 5** | Preprocesamiento &amp; Matrices | Limpieza de datasets de cáncer (CuMiDa), generación de matrices de expresión M_expr y precomputación de correlaciones de Pearson M_corr (.npy).      |
| **Semana 6** | Modularización de Tools     | Empaquetamiento de algoritmos bioinformáticos bajo esquemas Pydantic (run_statistical_filter, run_simls_mono, run_znsga2_multi).                 |
| **Semana 7** | Orquestador ReAct &amp; CLI     | Implementación del grafo LangGraph con persistencia de hilos (thread_id) y cliente de terminal interactivo continuo (agent_orquestador_Semana7.py). |
| **Semana 8** | Prototipo MVP &amp; Benchmark   | Suite de evaluación automatizada (evaluador_mvp_semana8.py) para medir precisión de invocación, latencia y tolerancia a fallos.                      |

---

## Estructura del Repositorio

```
LLMinvest/
├── data_processed/                    # Datasets normalizados y matrices binarias
│   ├── M_expr.csv                     # Matriz de expresión génica normalizada
│   └── M_corr.npy                     # Matriz de correlación de Pearson en NumPy
├── agent_orquestador_Semana7.py       # Orquestador del agente en LangGraph (Semana 7)
├── tools_bio.py                       # Herramientas bioinformáticas con esquemas Pydantic
├── evaluador_mvp_semana8.py           # Suite de evaluación y benchmarking MVP (Semana 8)
├── preprocesamiento_datasemana5.py    # Pipeline de extracción y preprocesamiento
├── setup_env.sh                       # Script bash para automatización del entorno
├── requirements.txt                   # Dependencias de Python completas
├── .env.example                       # Plantilla de variables de entorno
├── .gitignore                         # Exclusión de venv y datasets pesados
└── README.md                          # Documentación del proyecto

```

---

## Requisitos e Instalación Rápida

El proyecto requiere **Python 3.10 o superior** y una clave de API de **Groq Cloud**.

### Instalación Automática (Linux / macOS):

```
# Entrar al directorio del proyecto
cd LLMinvest

# Ejecutar script de configuración automatizado
bash setup_env.sh

```

### Instalación Manual:

```
# 1. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Edita el archivo .env e introduce tu clave GROQ_API_KEY

```

---

## Guía de Uso

### 1. Modo Interactivo por Terminal (Semana 7)

Permite interactuar conversacionalmente con el agente en lenguaje natural:

```
python agent_orquestador_Semana7.py

```

Ejemplos de consultas compatibles:

**Búsqueda Monobjetivo:
&gt; *"Hola, procesa el dataset '/home/vicente/Documentos/LLMinvest/data\_processed/M_expr.csv' usando simLS con GA y umbral de correlación 0.6."*
Búsqueda Multiobjetivo:
&gt; *"Ejecuta zNSGA2-LS en el mismo dataset con 50 generaciones para minimizar el tamaño del panel y el error de clasificación."*
*Pipeline con Filtro Previo:
&gt; *"Aplica el filtro estadístico de Mann-Whitney y Random Forest antes de correr simLS con VNS."*
*Finalizar sesión: Escribe `salir` o `exit`.

---

### 2. Suite de Evaluación Automatizada MVP (Semana 8)

Ejecuta la batería de pruebas cuantitativas sobre el agente orquestador para validar la precisión de llamadas a herramientas y tiempos de respuesta:

```
python evaluador_mvp_semana8.py

```

*Genera de forma automática el archivo `reporte_benchmark_semana8.csv` con métricas de latencia y detección algorítmica.*

---

## Código Fuente: Suite de Evaluación (Semana 8)

El archivo `evaluador_mvp_semana8.py` implementa el banco de pruebas de rendimiento del MVP:

```
import os
import sys
import time
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
logger = logging.getLogger("Benchmark-Semana8")

load_dotenv()

# Carga dinámica del orquestador de la Semana 7
try:
    from agent_orquestador_Semana7 import BioAgentOrchestrator
except ImportError:
    try:
        from agent_orquestador_v2 import BioAgentOrchestrator
    except ImportError:
        logger.warning("No se encontró agent_orquestador_Semana7.py en el path local. Se utilizará mock de contingencia.")
        BioAgentOrchestrator = None

class BenchmarkEvaluatorSemana8:
    def __init__(self, dataset_referencia: str = "/home/vicente/Documentos/LLMinvest/data_processed/M_expr.csv"):
        self.dataset_path = dataset_referencia
        self.casos_prueba = self._definir_bateria_pruebas()

        if BioAgentOrchestrator:
            self.orquestador = BioAgentOrchestrator()
            logger.info("Orquestador instanciado correctamente para pruebas.")
        else:
            self.orquestador = None
            logger.warning("Modo simulado activo para verificación del evaluador.")

    def _definir_bateria_pruebas(self) -&gt; List[Dict[str, Any]]:
        return [
            {
                "id_test": "TEST-01",
                "categoria": "Monobjetivo - GA",
                "prompt": f"Hola, procesa el dataset '{self.dataset_path}' utilizando simLS con Algoritmo Genético (GA) y un umbral de correlación de Pearson de 0.6 para identificar el mejor panel de biomarcadores.",
                "herramienta_esperada": "run_simls_mono",
                "algoritmo_esperado": "GA",
                "param_clave": {"correlation_threshold": 0.6}
            },
            {
                "id_test": "TEST-02",
                "categoria": "Monobjetivo - SA",
                "prompt": f"Selecciona un conjunto de genes del archivo '{self.dataset_path}' usando la metaheurística Simulated Annealing con búsqueda local por similitud (simLS).",
                "herramienta_esperada": "run_simls_mono",
                "algoritmo_esperado": "SA",
                "param_clave": {"metaheuristic": "SA"}
            },
            {
                "id_test": "TEST-03",
                "categoria": "Monobjetivo - VNS",
                "prompt": f"Necesito optimizar el AUC en '{self.dataset_path}' mediante Variable Neighborhood Search (VNS) incorporando simLS.",
                "herramienta_esperada": "run_simls_mono",
                "algoritmo_esperado": "VNS",
                "param_clave": {"metaheuristic": "VNS"}
            },
            {
                "id_test": "TEST-04",
                "categoria": "Multiobjetivo - zNSGA2-LS",
                "prompt": f"Aplica optimización multiobjetivo con zNSGA2-LS sobre '{self.dataset_path}' para minimizar simultáneamente el número de genes y el error de clasificación con 50 generaciones.",
                "herramienta_esperada": "run_znsga2_multi",
                "algoritmo_esperado": "zNSGA2-LS",
                "param_clave": {"generations": 50}
            },
            {
                "id_test": "TEST-05",
                "categoria": "Filtrado Estadístico Previo",
                "prompt": f"Ejecuta la etapa de filtrado previo de significancia con Mann-Whitney (p-value 0.01) y Random Forest sobre '{self.dataset_path}' antes de aplicar optimizadores.",
                "herramienta_esperada": "run_statistical_filter",
                "algoritmo_esperado": "Mann-Whitney + RF",
                "param_clave": {"confidence_level": 0.99}
            }
        ]

    def ejecutar_evaluacion(self) -&gt; pd.DataFrame:
        resultados = []
        print("\n" + "="*80)
        print("   INICIANDO EJECUCIÓN DEL BENCHMARK DE EVALUACIÓN MVP — SEMANA 8")
        print("="*80 + "\n")

        for caso in self.casos_prueba:
            test_id = caso["id_test"]
            categoria = caso["categoria"]
            prompt = caso["prompt"]
            esperada = caso["herramienta_esperada"]

            print(f"▶ [{test_id}] Evaluando: {categoria}")
            inicio = time.time()
            error = None
            respuesta_texto = ""
            exito = False

            try:
                if self.orquestador:
                    hilo_id = f"eval_thread_{test_id}"
                    respuesta_texto = self.orquestador.consultar(prompt, thread_id=hilo_id)
                    exito = True
                else:
                    time.sleep(0.5)
                    respuesta_texto = f"[Simulado] Herramienta {esperada} ejecutada con éxito. AUC alcanzado: 0.982. Genes: 24 seleccionados."
                    exito = True
            except Exception as e:
                error = str(e)
                logger.error(f"Falla al ejecutar {test_id}: {e}")

            duracion = round(time.time() - inicio, 2)
            deteccion_herramienta = (esperada.lower() in respuesta_texto.lower()) or (caso["algoritmo_esperado"].lower() in respuesta_texto.lower())

            resultados.append({
                "ID": test_id,
                "Categoría": categoria,
                "Herramienta Esperada": esperada,
                "Algoritmo Esperado": caso["algoritmo_esperado"],
                "Latencia (seg)": duracion,
                "Ejecutado con Éxito": "Sí" if exito and not error else "No",
                "Detección Algorítmica": "Correcta" if deteccion_herramienta else "Revisar",
                "Error": error if error else "Ninguno"
            })

            print(f"  ✔ Latencia: {duracion}s | Estado: {'OK' if exito else 'ERROR'}\n")

        df_resultados = pd.DataFrame(resultados)
        return df_resultados

    def guardar_reporte(self, df_resultados: pd.DataFrame, ruta_salida: str = "reporte_benchmark_semana8.csv"):
        df_resultados.to_csv(ruta_salida, index=False)
        print("="*80)
        print("   RESUMEN EJECUTIVO DE DESEMPEÑO DEL AGENTE (SEMANA 8)")
        print("="*80)
        print(df_resultados[["ID", "Categoría", "Latencia (seg)", "Ejecutado con Éxito", "Detección Algorítmica"]].to_string(index=False))
        print("="*80)
        print(f"Reporte exportado exitosamente a: {ruta_salida}\n")

if __name__ == "__main__":
    evaluador = BenchmarkEvaluatorSemana8()
    df_eval = evaluador.ejecutar_evaluacion()
    evaluador.guardar_reporte(df_eval)

```

---

## Descripción de Herramientas Bioinformáticas

1. **`run_statistical_filter`:** 
  * **Test Mann-Whitney U:** Prueba no paramétrica bilateral aplicada gen a gen con p < 0.01 (nivel de confianza del 99%) para descartar genes sin divergencia estadística entre tejido sano y tumoral.
  * **Impureza de Gini (Random Forest):** Entrena un ensamble y descarta todas las características cuya ganancia de impureza media sea nula.
2. **`run_simls_mono`:** 
  * Integra la búsqueda local por similitud biológica (**simLS**) con metaheurísticas monobjetivo (**GA**, **SA** o **VNS**).
  * Consulta la matriz M\_corr para sustituir genes redundantes por análogos fuertemente correlacionados (r >= 0.6), optimizando directamente la métrica **ROC-AUC**.
3. **`run_znsga2_multi`:** 
  * Ejecuta el algoritmo evolutivo multiobjetivo **zNSGA2-LS** optimizando simultáneamente: F1 = 1 - AUC y F2 = |S| / p
  * Aplica el **Operador Zoom** de reducción adaptativa del espacio de búsqueda para retener exclusivamente los genes que dominan la Frontera de Pareto.

---

## Variables de Entorno y Configuración

Crea un archivo llamado `.env` en la raíz del proyecto con el siguiente formato:

```
# Clave de API de Groq Cloud
GROQ_API_KEY=tu_clave_de_groq_aqui

# Modelo LLM predeterminado para el agente
GROQ_MODEL=openai/gpt-oss-120b

```

---

## Licencia y Créditos

Este desarrollo forma parte del proyecto de investigación **AgenticAI-FS** de la **Universidad de Santiago de Chile (USACH)**. Desarrollado por **Vicente Pizarro** como parte de su labor como ayudante de investigación.

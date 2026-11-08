"""
AGENTICAI-FS — SEMANA 7: AGENTE ORQUESTADOR (CLI INTERACTIVO - v2)
==================================================================
Agente Autónomo Bioinformático basado en LangGraph / LangChain ReAct.
Permite la interacción dinámica mediante consola/terminal con input() en tiempo real.
Mantiene la memoria conversacional de estado basada en hilos (thread_id).
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ------------------------------------------------------------------------------
# 1. CARGA DE CONFIGURACIÓN Y RUTAS
# ------------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("AgenticAI-FS-Orchestrator")

# Importar las herramientas definidas en tools_metaheuristicassemana6.py (Semana 6)
try:
    from tools_metaheuristicassemana6 import BIO_TOOLS
    logger.info(f"Se cargaron {len(BIO_TOOLS)} herramientas bioinformáticas exitosamente.")
except ImportError:
    logger.warning("No se pudo importar 'tools_metaheuristicassemana6.py'. Verifica que el archivo esté en el mismo directorio.")
    BIO_TOOLS = []

# ------------------------------------------------------------------------------
# 2. PROMPT DEL SISTEMA PARA EL ORQUESTADOR REACT
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Eres AgenticAI-FS, un agente autónomo bioinformático experto en selección de características (Feature Selection) para datos de expresión génica (RNA-seq y microarrays).

TU OBJETIVO:
Ayudar a investigadores y profesionales médicos a identificar paneles compactos de genes (biomarcadores) con alta capacidad predictiva (ROC-AUC) utilizando metaheurísticas híbridas.

DISPONES DE LAS SIGUIENTES HERRAMIENTAS:
1. `run_statistical_filter`: Filtro inicial (Mann-Whitney + Impureza de Random Forest). Úsalo cuando se requiera reducir preliminarmente la dimensionalidad de un dataset masivo.
2. `run_simls_mono`: Metaheurística Monobjetivo con búsqueda local por similitud (simLS + GA, SA o VNS). Ideal para maximizar el AUC-ROC y obtener un panel de manera rápida.
3. `run_znsga2_multi`: Algoritmo Evolutivo Multiobjetivo (zNSGA2-LS con operador Zoom). Úsalo cuando el usuario busque equilibrar la reducción de genes y la tasa de error (Frontera de Pareto).

INSTRUCCIONES DE RAZONAMIENTO:
- Analiza la consulta del usuario para identificar el dataset, la metaheurística deseada y las restricciones (ej. AUC mínimo, número máximo de genes).
- Selecciona la herramienta adecuada y construye los parámetros requeridos.
- Si la consulta es ambigua respecto al algoritmo, sugiere 'zNSGA2-LS' para análisis multiobjetivo o 'simLS + GA' para monobjetivo.
- Cuando la herramienta responda, sintetiza los resultados explicando:
  1. Algoritmo y configuración utilizada.
  2. Métrica ROC-AUC alcanzada.
  3. Cantidad y lista de genes seleccionados.
  4. Si se cumplieron las restricciones planteadas.
- Responde siempre en español con un tono profesional, claro y científicamente riguroso.
"""

# ------------------------------------------------------------------------------
# 3. INICIALIZACIÓN DEL LLM Y MEMORIA DE LANGGRAPH
# ------------------------------------------------------------------------------
def obtener_llm() -> ChatGroq:
    """Inicializa el cliente de Groq (GPT-OSS / Llama 3) para el agente."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Error: La variable de entorno GROQ_API_KEY no está configurada en el archivo .env")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,  # Temperatura baja para ejecuciones deterministas de tools
        reasoning_effort="medium",
        reasoning_format="hidden",
        streaming=False,
        model_kwargs={"max_completion_tokens": 4096}
    )

# Checkpointer en memoria para persistencia de la conversación por thread_id
checkpointer_memoria = MemorySaver()

def construir_agente():
    """Crea el grafo del agente ReAct combinando el LLM, las herramientas y la memoria."""
    llm = obtener_llm()
    
    agente_grafo = create_react_agent(
        model=llm,
        tools=BIO_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer_memoria
    )
    return agente_grafo

# ------------------------------------------------------------------------------
# 4. CLASE PRINCIPAL DEL AGENTE ORQUESTADOR
# ------------------------------------------------------------------------------
class BioAgentOrchestrator:
    def __init__(self):
        self.agente = construir_agente()
        logger.info("Agente Orquestador AgenticAI-FS (Semana 7 MVP) listo para operar.")

    def consultar(self, mensaje_usuario: str, thread_id: str = "sesion_defecto") -> str:
        """
        Procesa una consulta del usuario dentro de un hilo de conversación.
        Mantiene el contexto conversacional a través de thread_id.
        """
        config = {"configurable": {"thread_id": thread_id}}
        entrada = {"messages": [HumanMessage(content=mensaje_usuario)]}

        logger.info(f"Ejecutando consulta en el hilo '{thread_id}'...")

        # Invocar el grafo de LangGraph
        resultado = self.agente.invoke(entrada, config=config)

        # Retornar la respuesta final del LLM
        respuesta_final = resultado["messages"][-1].content
        return respuesta_final

# ------------------------------------------------------------------------------
# 5. MODO INTERACTIVO POR TERMINAL (CLI)
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    try:
        orquestador = BioAgentOrchestrator()
        THREAD_SESION = "sesion_terminal_01"

        print("Ingresa tu consulta para AgenticAI-FS (escribe 'salir' o 'exit' para terminar):\n")

        while True:
            # Captura de entrada del usuario desde la terminal
            usuario_input = input("[Usuario]: ").strip()

            # Condición de salida del bucle
            if usuario_input.lower() in ["salir", "exit", "quit"]:
                print("\n¡Sesión finalizada. Hasta luego!")
                break

            # Evitar enviar consultas vacías
            if not usuario_input:
                continue

            # Procesar consulta con el agente manteniendo el thread_id
            respuesta = orquestador.consultar(usuario_input, thread_id=THREAD_SESION)
            print(f"\n[AgenticAI-FS]:\n{respuesta}\n")
            print("-" * 75 + "\n")

    except Exception as err:
        logger.error(f"Error al ejecutar el agente: {err}")

#!/bin/bash
# Script de automatización de entorno virtual e instalación para AgenticAI-FS

echo "=========================================================================="
echo "      AGENTICAI-FS — CONFIGURACIÓN AUTOMÁTICA DEL ENTORNO VIRTUAL"
echo "=========================================================================="

# 1. Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual 'venv'..."
    python3 -m venv venv
else
    echo "El entorno virtual 'venv' ya existe."
fi

# 2. Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# 3. Actualizar pip e instalar dependencias
echo "Actualizando pip e instalando dependencias desde requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=========================================================================="
echo "¡Instalación completada exitosamente!"
echo "Para activar el entorno en tu terminal ejecuta:"
echo "    source venv/bin/activate"
echo "Para correr el agente interactivo:"
echo "    python agent_orquestador_Semana7.py"
echo "=========================================================================="

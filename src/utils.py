import yaml
import joblib
import logging
from pathlib import Path
import os

def setup_logging():
    """Configura el logging para el proyecto"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path='../config/model_params.yaml'):  # Cambiado a ruta relativa
    """
    Carga la configuración desde un archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Configuración cargada
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si el archivo no es un YAML válido
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error al cargar configuración: {e}")
        raise

def save_model(model, model_path):
    """Guarda el modelo en la ruta especificada"""
    ensure_dir(os.path.dirname(model_path))
    joblib.dump(model, model_path) 

def load_model(model_path):
    """Carga un modelo guardado"""
    return joblib.load(model_path)

def ensure_dir(dir_path):
    """Asegura que un directorio existe"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
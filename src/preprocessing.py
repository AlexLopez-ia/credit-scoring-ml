import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from src.utils import ensure_dir

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, path):
        """Carga los datos desde el archivo CSV"""
        try:
            df = pd.read_csv(path)
            self.logger.info(f"Datos cargados exitosamente de {path}")
            return df
        except Exception as e:
            self.logger.error(f"Error al cargar los datos: {e}")
            raise
            
    def handle_missing_values(self, df):
        """Maneja los valores faltantes en el dataset"""
        for col in self.config['columns_to_impute']:
            df[col] = df[col].fillna(df[col].median())
        return df
    
    def handle_outliers(self, df, columns, n_std=3):
        """Detecta y maneja outliers"""
        df_clean = df.copy()
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            df_clean[column] = df_clean[column].clip(
                lower=mean - n_std * std,
                upper=mean + n_std * std
            )
        return df_clean
    
    def scale_features(self, df, columns_to_scale):
        """Escala las características seleccionadas"""
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df_scaled 
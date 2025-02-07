import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import Dict, List

class FeatureEngineer:
    def __init__(self, config: Dict):
        """
        Inicializa el FeatureEngineer.
        
        Args:
            config (Dict): Configuración con parámetros para ingeniería de características
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Valida que el DataFrame tenga las columnas necesarias.
        
        Args:
            df (pd.DataFrame): DataFrame a validar
        
        Raises:
            ValueError: Si faltan columnas requeridas
        """
        required_columns = [
            'DebtRatio', 
            'MonthlyIncome',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas características a partir del DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con nuevas características
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Crear características
            df['DebtToIncome'] = df['DebtRatio'] * df['MonthlyIncome']
            df['TotalDelinquencies'] = (
                df['NumberOfTime30-59DaysPastDueNotWorse'] +
                df['NumberOfTime60-89DaysPastDueNotWorse'] +
                df['NumberOfTimes90DaysLate']
            )
            
            self.logger.info("Características creadas exitosamente")
            return df
            
        except Exception as e:
            self.logger.error(f"Error en create_features: {e}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 'all') -> pd.DataFrame:
        """
        Selecciona las características más importantes.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Variable objetivo
            k (int): Número de características a seleccionar
            
        Returns:
            pd.DataFrame: DataFrame con scores de características
        """
        try:
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            
            feature_scores = pd.DataFrame({
                'Feature': X.columns,
                'Score': selector.scores_,
                'P-value': selector.pvalues_
            })
            
            return feature_scores.sort_values('Score', ascending=False)
        except Exception as e:
            self.logger.error(f"Error en select_features: {e}")
            raise 
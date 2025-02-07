import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer

def test_create_features():
    # Crear datos de prueba
    test_data = pd.DataFrame({
        'DebtRatio': [0.5, 0.3, 0.7],
        'MonthlyIncome': [5000, 3000, 7000],
        'NumberOfTime30-59DaysPastDueNotWorse': [0, 1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 1],
        'NumberOfTimes90DaysLate': [0, 0, 0]
    })
    
    config = {
        'create_features': ['DebtToIncome', 'TotalDelinquencies']
    }
    
    engineer = FeatureEngineer(config)
    result = engineer.create_features(test_data)
    
    # Verificar que se crearon las nuevas características
    assert 'DebtToIncome' in result.columns
    assert 'TotalDelinquencies' in result.columns
    
    # Verificar cálculos
    assert result['DebtToIncome'].iloc[0] == test_data['DebtRatio'].iloc[0] * test_data['MonthlyIncome'].iloc[0]
    assert result['TotalDelinquencies'].iloc[1] == 1

def test_validate_dataframe():
    engineer = FeatureEngineer({})
    
    # DataFrame sin columnas requeridas
    invalid_df = pd.DataFrame({
        'OtraColumna': [1, 2, 3]
    })
    
    with pytest.raises(ValueError):
        engineer.validate_dataframe(invalid_df) 
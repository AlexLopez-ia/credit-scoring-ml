import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

def test_handle_missing_values():
    # Crear datos de prueba
    test_data = pd.DataFrame({
        'MonthlyIncome': [1000, None, 3000],
        'NumberOfDependents': [2, None, 1]
    })
    
    config = {
        'columns_to_impute': ['MonthlyIncome', 'NumberOfDependents']
    }
    
    preprocessor = DataPreprocessor(config)
    result = preprocessor.handle_missing_values(test_data)
    
    assert result['MonthlyIncome'].isna().sum() == 0
    assert result['NumberOfDependents'].isna().sum() == 0 

def test_handle_outliers():
    test_data = pd.DataFrame({
        'MonthlyIncome': [1000, 5000, 1000000],  # Con outlier
    })
    
    config = {
        'outlier_columns': ['MonthlyIncome'],
        'outlier_std': 3
    }
    
    preprocessor = DataPreprocessor(config)
    result = preprocessor.handle_outliers(test_data, ['MonthlyIncome'])
    
    # Calcular el límite superior esperado
    mean = test_data['MonthlyIncome'].mean()
    std = test_data['MonthlyIncome'].std()
    expected_upper = mean + 3 * std
    
    # Verificar que el outlier fue tratado
    assert result['MonthlyIncome'].max() <= expected_upper

def test_scale_features():
    test_data = pd.DataFrame({
        'age': [20, 30, 40],
        'MonthlyIncome': [1000, 2000, 3000]
    })
    
    config = {
        'columns_to_scale': ['age', 'MonthlyIncome']
    }
    
    preprocessor = DataPreprocessor(config)
    result = preprocessor.scale_features(test_data, config['columns_to_scale'])
    
    # Verificar que los datos están escalados (media ≈ 0, std ≈ 1)
    assert abs(result['age'].mean()) < 1e-10  # Muy cerca de 0
    assert abs(result['age'].std(ddof=0) - 1) < 1e-10  # Usar ddof=0 para población
    
    # También verificar MonthlyIncome
    assert abs(result['MonthlyIncome'].mean()) < 1e-10
    assert abs(result['MonthlyIncome'].std(ddof=0) - 1) < 1e-10

def test_data_preprocessor():
    # Configuración de prueba
    config = {
        'columns_to_impute': ['MonthlyIncome', 'NumberOfDependents'],
        'columns_to_scale': ['age', 'MonthlyIncome', 'DebtRatio']
    }
    
    # Crear instancia del preprocesador
    preprocessor = DataPreprocessor(config)
    
    # Aquí tus tests... 
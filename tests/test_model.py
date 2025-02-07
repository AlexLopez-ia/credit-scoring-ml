import pytest
import numpy as np
from src.model import CreditScoringModel

def test_model_initialization():
    params = {
        'penalty': 'l2',
        'solver': 'lbfgs'
    }
    model = CreditScoringModel(params)
    assert model.params == params

def test_model_fit_predict():
    # Crear datos sintéticos
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    model = CreditScoringModel()
    model.fit(X, y)

    # Verificar predicciones
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    # Modificamos la verificación para aceptar numpy.int32 y numpy.int64
    assert all(isinstance(pred, (np.integer, int)) for pred in predictions)
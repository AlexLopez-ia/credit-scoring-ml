from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import logging

class CreditScoringModel(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        self.params = params or {}
        self.model = LogisticRegression(**self.params)
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X, y):
        """Entrena el modelo"""
        self.logger.info("Iniciando entrenamiento del modelo")
        self.model.fit(X, y)
        self.logger.info("Entrenamiento completado")
        return self
    
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evalúa el modelo"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        print("\nReporte de Clasificación:")
        print(classification_report(y, y_pred))
        
        auc_roc = roc_auc_score(y, y_pred_proba)
        print(f"\nAUC-ROC: {auc_roc:.3f}")
        
        return {
            'auc_roc': auc_roc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        } 
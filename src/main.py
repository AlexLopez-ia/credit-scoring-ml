import logging
from pathlib import Path
from src.utils import load_config, setup_logging, save_model
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import CreditScoringModel
from sklearn.model_selection import train_test_split


def main():
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Cargar configuración
        config = load_config('config/model_params.yaml')  # Quitamos el '../'
        
        # Inicializar componentes
        preprocessor = DataPreprocessor(config['preprocessing'])
        feature_engineer = FeatureEngineer(config['feature_engineering'])
        model = CreditScoringModel(config['model']['logistic_regression'])
        
        # Ejecutar pipeline
        logger.info("Iniciando pipeline de ML")
        
        # 1. Preprocesamiento
        df = preprocessor.load_data('data/raw/cs-training.csv')  # Quitamos el '../'
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.handle_outliers(df, config['preprocessing']['outlier_columns'])
        

        # 2. Feature Engineering
        df = feature_engineer.create_features(df)
        
        # 3. Entrenamiento y evaluación
        # Separar features y target
        X = df.drop('SeriousDlqin2yrs', axis=1)
        y = df['SeriousDlqin2yrs']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenamiento
        model.fit(X_train, y_train)
        
        # Evaluación
        results = model.evaluate(X_test, y_test)
        
        # Guardar modelo
        save_model(model, 'models/credit_scoring_model.joblib')  # Quitamos el '../'
        
        logger.info("Pipeline completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 
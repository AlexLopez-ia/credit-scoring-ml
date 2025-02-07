# Credit Scoring ML Project

Modelo de machine learning para predecir el riesgo crediticio de clientes basado en sus historiales financieros.

## 📊 Descripción
Este proyecto implementa un modelo de clasificación para predecir la probabilidad de impago de créditos. Utiliza características como el ratio de deuda, ingresos mensuales y el historial de pagos atrasados para realizar las predicciones.

Este es un proyecto de práctica personal utilizando el dataset público ["Give Me Some Credit"](https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv) de Kaggle.

## 🛠 Tecnologías Utilizadas
- Python 3.10
- scikit-learn
- pandas
- numpy
- pytest

## 📁 Estructura del Proyecto
```
prediction_scoring/
├── config/          # Configuración del modelo
│   └── model_params.yaml
├── data/            # Datos (no incluidos en git)
│   └── raw/
├── models/          # Modelos entrenados
├── src/             # Código fuente
│   ├── main.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── utils.py
└── tests/           # Tests unitarios
```

## 🚀 Instalación y Uso

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd prediction_scoring
```

2. Descargar el dataset:
- Descargar el archivo `cs-training.csv` del [repositorio original](https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv)
- Colocarlo en la carpeta `data/raw/`

3. Crear y activar entorno virtual:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/Mac
```

4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

5. Ejecutar el modelo:
```bash
python -m src.main
```

6. Ejecutar tests:
```bash
python -m pytest tests/
```

## 📈 Resultados
- AUC-ROC Score: 0.805
- Precisión para clase positiva (default): 0.26
- Recall: 0.67

## ✅ Tests
El proyecto incluye tests unitarios para:
- Preprocesamiento de datos
- Ingeniería de características
- Validación del modelo

## 🤝 Contribuciones
Este es un proyecto de práctica personal para aprendizaje de machine learning y buenas prácticas de desarrollo. El dataset utilizado es público y se usa solo con fines educativos.

## 📝 Licencia
[MIT License](https://opensource.org/licenses/MIT)

## 👤 Autor
[Alex López]

## 📚 Dataset
Dataset original: ["Give Me Some Credit"](https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv) de Kaggle

```yaml
preprocessing:
  columns_to_impute:
    - MonthlyIncome 


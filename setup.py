from setuptools import setup, find_packages

setup(
    name="credit-scoring-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pyyaml>=5.4.0',
        'joblib>=1.0.0',
    ],
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Proyecto de machine learning para predicción de riesgo crediticio",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/credit-scoring-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 
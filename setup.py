from setuptools import setup, find_packages

setup(
    name="amazon-reviews",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=2.1.0',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'nltk>=3.8.1',
        'sentence-transformers>=2.5.0',
        'torch>=2.0.1',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'kaggle>=1.5.16',
        'kagglehub>=0.1.4',
        'wordcloud>=1.9.2',
        'behave>=1.2.6'
    ]
) 
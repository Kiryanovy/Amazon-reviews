# Sistema de Análise de Sentimentos de Reviews da Amazon

Este projeto implementa um sistema robusto de análise de sentimentos para avaliações de produtos da Amazon, apresentando múltiplos modelos de machine learning e capacidades abrangentes de processamento de texto.

## Configuração Inicial

1. Clone o repositório:
```bash
git clone [repository-url]
cd amazon-reviews
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure suas credenciais do Kaggle:
   - Acesse sua conta no [Kaggle](https://www.kaggle.com/)
   - Vá em "Account" > "Create New API Token"
   - Baixe o arquivo `kaggle.json`
   - Crie o diretório para as credenciais:
     ```bash
     # No Windows:
     mkdir %USERPROFILE%\.kaggle
     copy kaggle.json %USERPROFILE%\.kaggle\
     
     # No Linux/Mac:
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     ```
   - Ajuste as permissões (Linux/Mac):
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

5. Download inicial dos dados:
```bash
# Cria diretório para os dados
mkdir data

# Download do dataset
python -c "from src.data_loading import load_amazon_reviews; load_amazon_reviews()"
```

6. Execute o treinamento inicial dos modelos:
```bash
python src/modeling.py
```

## Funcionalidades

- Múltiplas implementações de modelos (SVM, Random Forest, Rede Neural)
- Pipeline avançado de pré-processamento de texto
- Suporte para embeddings baseados em transformers e métodos tradicionais
- Métricas de avaliação abrangentes
- Capacidade de simulação de reviews
- Funcionalidade de persistência de modelos (salvar/carregar)

## Estrutura do Projeto

```
amazon-reviews/
├── src/
│   ├── data_loading.py    # Carregamento e preparação dos dados
│   ├── preprocessing.py    # Pipeline de pré-processamento de texto
│   ├── feature_extraction.py # Métodos de extração de características
│   ├── modeling.py        # Implementação e treinamento dos modelos
├── models/                # Arquivos dos modelos salvos
├── requirements.txt       # Dependências do projeto
└── README.md
```

## Desempenho dos Modelos

O sistema inclui múltiplos modelos com as seguintes métricas de desempenho:

- SVM com Transformer: 84,00% de acurácia
- SVM com TF-IDF/SVD: 81,00% de acurácia
- Random Forest com Transformer: 78,30% de acurácia
- MLP com Transformer: 81,15% de acurácia

## Como Usar

### Treinando os Modelos

```python
from src.modeling import train_and_evaluate_models
from src.data_loading import load_amazon_reviews
from src.preprocessing import preprocess_dataset

# Carrega e pré-processa os dados
df = load_amazon_reviews(sample_size=10000)  # Ajuste o tamanho da amostra conforme necessário
processed_df = preprocess_dataset(df)

# Configura os modelos
models_config = [
    {'type': 'svm', 'use_transformer': True},
    {'type': 'svm', 'use_transformer': False},
    {'type': 'rf', 'use_transformer': True},
    {'type': 'mlp', 'use_transformer': True}
]

# Treina e avalia
results = train_and_evaluate_models(processed_df, models_config)
```

### Simulando Reviews

```python
from src.modeling import simulate_reviews

# Testa um novo review
test_review = "Este produto é incrível! Eu amei tudo sobre ele."
predictions = simulate_reviews(test_review)

# Imprime as previsões
for model, sentiment in predictions.items():
    print(f"{model}: {sentiment}")
```

## Dependências

- Python 3.x
- NumPy (>=2.1.0)
- Pandas (>=2.0.3)
- Scikit-learn (>=1.3.0)
- NLTK (>=3.8.1)
- Sentence-Transformers (>=2.5.0)
- PyTorch (>=2.0.1)
- Dependências adicionais em requirements.txt

## Detalhes das Funcionalidades

### Pré-processamento de Texto
- Conversão para minúsculas
- Remoção de URLs e e-mails
- Tratamento de pontuação
- Tokenização
- Remoção de stopwords
- Lematização
- Filtragem por tamanho de token

### Extração de Características
- Vetorização TF-IDF
- Embeddings baseados em transformers
- Redução de dimensionalidade usando TruncatedSVD

### Opções de Modelos
- Support Vector Machine (SVM)
- Random Forest
- Rede Neural Multi-camada (MLP) 
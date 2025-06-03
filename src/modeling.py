import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
from src.data_loading import load_amazon_reviews
from src.preprocessing import preprocess_dataset
from src.feature_extraction import FeatureExtractor
import pickle
import os

class SentimentClassifier:
    def __init__(self, model_type='svm', use_transformer=True, transformer_model='all-MiniLM-L6-v2'):
        """
        Inicializa o classificador de sentimentos
        Args:
            model_type: 'svm', 'rf' (Random Forest) ou 'mlp' (Neural Network)
            use_transformer: Se True, usa SentenceTransformer para embeddings
            transformer_model: Nome do modelo transformer a ser usado
        """
        self.model_type = model_type
        self.use_transformer = use_transformer
        self.transformer_model = transformer_model
        
        # Inicializa o modelo tradicional
        if model_type == 'svm':
            self.model = LinearSVC(random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        
        # Inicializa o transformer se necessário
        self.transformer = None
        if use_transformer:
            print(f"Carregando modelo {transformer_model}...")
            self.transformer = SentenceTransformer(transformer_model)
        
        # Inicializa o extrator de features tradicional
        self.feature_extractor = FeatureExtractor()
        
        # Inicializa o scaler para normalização
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False para matrizes esparsas
    
    def get_embeddings(self, texts):
        """Obtém embeddings usando o transformer ou feature extractor"""
        if self.use_transformer:
            # Converte Series para lista
            texts_list = texts.tolist() if isinstance(texts, pd.Series) else texts
            return self.transformer.encode(texts_list, show_progress_bar=True)
        else:
            # Cria um DataFrame temporário para usar com o feature extractor
            temp_df = pd.DataFrame({'processed_text': texts})
            
            # Treina o TF-IDF se necessário
            if not hasattr(self.feature_extractor.tfidf, 'vocabulary_'):
                self.feature_extractor.fit_tfidf(temp_df['processed_text'])
            
            # Obtém a matriz TF-IDF
            tfidf_matrix = self.feature_extractor.transform_tfidf(temp_df['processed_text'])
            
            # Treina o embedder (TruncatedSVD) se necessário
            if not hasattr(self.feature_extractor.embedder, 'components_'):
                self.feature_extractor.train_embeddings(temp_df['processed_text'])
            
            # Retorna os embeddings
            return self.feature_extractor.transform_embeddings(temp_df['processed_text'])
    
    def train(self, X_train, y_train):
        """Treina o modelo"""
        print(f"Treinando modelo {self.model_type}...")
        
        # Normaliza os dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Treina o modelo
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        """Faz previsões"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        y_pred = self.predict(X_test)
        
        # Calcula métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, output_dir='models'):
        """Salva o modelo"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_path = os.path.join(output_dir, f'sentiment_classifier_{self.model_type}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_extractor': self.feature_extractor,
                'model_type': self.model_type,
                'use_transformer': self.use_transformer,
                'transformer_model': self.transformer_model
            }, f)
    
    def load_model(self, output_dir='models'):
        """Carrega o modelo salvo"""
        model_path = os.path.join(output_dir, f'sentiment_classifier_{self.model_type}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_extractor = data['feature_extractor']
                self.model_type = data['model_type']
                self.use_transformer = data['use_transformer']
                self.transformer_model = data['transformer_model']
                if self.use_transformer:
                    self.transformer = SentenceTransformer(self.transformer_model)

def train_and_evaluate_models(df, models_config):
    """
    Treina e avalia múltiplos modelos
    Args:
        df: DataFrame com os dados
        models_config: Lista de configurações de modelos para testar
    Returns:
        dict: Resultados da avaliação de cada modelo
    """
    # Prepara os dados
    X = df['processed_text'].values  # Convertendo para array numpy
    y = df['sentiment'].values
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Treina e avalia cada modelo
    for config in models_config:
        model_name = f"{config['type']}_{'transformer' if config['use_transformer'] else 'traditional'}"
        print(f"\nTreinando modelo: {model_name}")
        
        # Inicializa o classificador
        classifier = SentimentClassifier(
            model_type=config['type'],
            use_transformer=config['use_transformer']
        )
        
        # Obtém embeddings
        X_train_emb = classifier.get_embeddings(X_train)
        X_test_emb = classifier.get_embeddings(X_test)
        
        # Treina o modelo
        classifier.train(X_train_emb, y_train)
        
        # Avalia o modelo
        metrics = classifier.evaluate(X_test_emb, y_test)
        results[model_name] = metrics
        
        # Salva o modelo
        classifier.save_model()
        
        print(f"Resultados para {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return results

def simulate_reviews(text, models_dir='models'):
    """
    Simula a classificação de novos reviews usando todos os modelos disponíveis
    Args:
        text: Texto do review para classificar
        models_dir: Diretório com os modelos salvos
    Returns:
        dict: Classificações de cada modelo
    """
    results = {}
    
    # Lista todos os modelos salvos
    for file in os.listdir(models_dir):
        if file.startswith('sentiment_classifier_'):
            model_type = file.split('_')[2].split('.')[0]
            
            # Carrega o modelo
            classifier = SentimentClassifier(model_type=model_type)
            classifier.load_model()
            
            # Pré-processa o texto
            processed_text = preprocess_dataset(pd.DataFrame({'text': [text]}))['processed_text'].iloc[0]
            
            # Obtém embeddings
            embeddings = classifier.get_embeddings([processed_text])
            
            # Faz a previsão
            prediction = classifier.predict(embeddings)[0]
            sentiment = "Positivo" if prediction == 2 else "Negativo"
            
            results[f"Modelo {model_type}"] = sentiment
    
    return results

if __name__ == "__main__":
    # Carrega e pré-processa os dados
    print("Carregando e pré-processando dados...")
    df = load_amazon_reviews(sample_size=10000)  # Usando amostra menor para teste
    processed_df = preprocess_dataset(df)
    
    # Configuração dos modelos para testar
    models_config = [
        {'type': 'svm', 'use_transformer': True},
        {'type': 'svm', 'use_transformer': False},
        {'type': 'rf', 'use_transformer': True},
        {'type': 'mlp', 'use_transformer': True}
    ]
    
    # Treina e avalia os modelos
    results = train_and_evaluate_models(processed_df, models_config)
    
    # Exemplo de simulação
    print("\nTestando simulação de reviews:")
    test_review = "Produto horrível, não recomendo."
    predictions = simulate_reviews(test_review)
    
    print("\nPrevisões para o review de teste:")
    for model, sentiment in predictions.items():
        print(f"{model}: {sentiment}") 
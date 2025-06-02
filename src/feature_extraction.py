import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from data_loading import load_amazon_reviews
from preprocessing import preprocess_dataset
import pickle
import os

class FeatureExtractor:
    def __init__(self, max_features=5000, embedding_size=100):
        """
        Inicializa o extrator de features
        Args:
            max_features: Número máximo de features para TF-IDF
            embedding_size: Dimensão dos vetores de embedding
        """
        self.max_features = max_features
        self.embedding_size = embedding_size
        
        # Inicializa os modelos
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=5,              # Ignora termos que aparecem em menos de 5 documentos
            max_df=0.8,           # Ignora termos que aparecem em mais de 80% dos documentos
            ngram_range=(1, 2)    # Considera unigramas e bigramas
        )
        
        # Substitui Word2Vec por TruncatedSVD para word embeddings
        self.embedder = TruncatedSVD(
            n_components=embedding_size,
            random_state=42
        )
        
    def fit_tfidf(self, texts):
        """Treina o modelo TF-IDF"""
        print("Treinando modelo TF-IDF...")
        self.tfidf.fit(texts)
        print(f"Vocabulário TF-IDF: {len(self.tfidf.vocabulary_)} termos")
        
    def transform_tfidf(self, texts):
        """Transforma textos em vetores TF-IDF"""
        return self.tfidf.transform(texts)
    
    def train_embeddings(self, texts):
        """
        Treina o modelo de embeddings usando TruncatedSVD sobre TF-IDF
        Args:
            texts: Lista de textos
        """
        print("Treinando modelo de embeddings...")
        # Primeiro aplica TF-IDF
        tfidf_matrix = self.tfidf.transform(texts)
        # Depois reduz dimensionalidade com TruncatedSVD
        self.embedder.fit(tfidf_matrix)
        print(f"Dimensões do embedding: {self.embedding_size}")
    
    def transform_embeddings(self, texts):
        """
        Transforma textos em vetores de embedding
        Args:
            texts: Lista de textos
        Returns:
            Matriz de embeddings
        """
        # Primeiro aplica TF-IDF
        tfidf_matrix = self.tfidf.transform(texts)
        # Depois transforma com TruncatedSVD
        return self.embedder.transform(tfidf_matrix)
    
    def save_models(self, output_dir='models'):
        """Salva os modelos treinados"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Salva TF-IDF e embedder
        with open(os.path.join(output_dir, 'feature_extractor.pkl'), 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'embedder': self.embedder
            }, f)
    
    def load_models(self, output_dir='models'):
        """Carrega os modelos salvos"""
        model_path = os.path.join(output_dir, 'feature_extractor.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
                self.tfidf = models['tfidf']
                self.embedder = models['embedder']

def extract_features(df, extractor=None, max_features=5000, embedding_size=100):
    """
    Extrai features usando TF-IDF e embeddings
    Args:
        df: DataFrame com os textos processados
        extractor: FeatureExtractor pré-treinado (opcional)
        max_features: Número máximo de features para TF-IDF
        embedding_size: Dimensão dos vetores de embedding
    Returns:
        features_tfidf: Matriz de features TF-IDF
        features_embeddings: Matriz de embeddings
        extractor: FeatureExtractor treinado
    """
    if extractor is None:
        extractor = FeatureExtractor(max_features=max_features, embedding_size=embedding_size)
        
        # Treina TF-IDF
        extractor.fit_tfidf(df['processed_text'])
        
        # Treina embeddings
        extractor.train_embeddings(df['processed_text'])
        
        # Salva os modelos
        extractor.save_models()
    
    # Extrai features
    features_tfidf = extractor.transform_tfidf(df['processed_text'])
    features_embeddings = extractor.transform_embeddings(df['processed_text'])
    
    return features_tfidf, features_embeddings, extractor

if __name__ == "__main__":
    # Carrega e pré-processa os dados
    print("Carregando e pré-processando dados...")
    df = load_amazon_reviews(sample_size=10000)  # Usando amostra menor para teste
    processed_df = preprocess_dataset(df)
    
    # Extrai features
    print("\nExtraindo features...")
    features_tfidf, features_embeddings, extractor = extract_features(processed_df)
    
    # Mostra informações sobre as features
    print("\nInformações das features:")
    print(f"TF-IDF shape: {features_tfidf.shape}")
    print(f"Embeddings shape: {features_embeddings.shape}")
    
    # Mostra exemplo de palavras mais importantes
    if len(processed_df) > 0:
        print("\nExemplo de documento:")
        idx = 0  # Primeiro documento
        
        print("\nTexto original:")
        print(processed_df.iloc[idx]['text'][:200], "...")
        
        print("\nPalavras mais importantes (TF-IDF):")
        feature_names = np.array(extractor.tfidf.get_feature_names_out())
        tfidf_scores = features_tfidf[idx].toarray()[0]
        top_indices = np.argsort(tfidf_scores)[-10:][::-1]
        for idx in top_indices:
            if tfidf_scores[idx] > 0:
                print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")
        
        print("\nVetor de embedding (primeiros 5 valores):")
        print(features_embeddings[idx][:5], "...") 
"""
Embedding extraction for Amazon Reviews sentiment analysis.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class EmbeddingExtractor:
    """Handles different types of text embeddings."""
    
    def __init__(self, embedding_type='transformer', transformer_model='all-MiniLM-L6-v2', 
                 max_features=5000, n_components=100):
        """
        Initialize embedding extractor.
        
        Args:
            embedding_type: 'transformer' or 'tfidf_svd'
            transformer_model: Name of the transformer model to use
            max_features: Max features for TF-IDF
            n_components: Number of components for SVD
        """
        self.embedding_type = embedding_type
        self.transformer_model = transformer_model
        self.max_features = max_features
        self.n_components = n_components
        
        # Initialize models
        if embedding_type == 'transformer':
            self.model = SentenceTransformer(transformer_model)
        else:
            self.tfidf = TfidfVectorizer(max_features=max_features)
            self.svd = TruncatedSVD(n_components=n_components)
    
    def fit(self, texts):
        """Fit the embedding model on texts."""
        if self.embedding_type != 'transformer':
            # Only TF-IDF+SVD needs fitting
            tfidf_matrix = self.tfidf.fit_transform(texts)
            self.svd.fit(tfidf_matrix)
    
    def transform(self, texts):
        """Transform texts to embeddings."""
        if self.embedding_type == 'transformer':
            return self.model.encode(texts, show_progress_bar=True)
        else:
            tfidf_matrix = self.tfidf.transform(texts)
            return self.svd.transform(tfidf_matrix)
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        if self.embedding_type == 'transformer':
            return self.transform(texts)
        else:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            self.svd.fit(tfidf_matrix)
            return self.svd.transform(tfidf_matrix)

if __name__ == "__main__":
    # Exemplo de uso
    print("Demonstração do EmbeddingExtractor:\n")
    
    # Textos de exemplo
    example_texts = [
        "This product is amazing! I love it so much. The quality is great.",
        "Terrible product. Don't waste your money. Poor quality and bad customer service!!!",
        "It's okay, nothing special. Average product for the price."
    ]
    
    # Testar ambos os tipos de embedding
    for embedding_type in ['transformer', 'tfidf_svd']:
        print(f"\nTestando {embedding_type.upper()}:")
        
        # Criar extrator com dimensão menor para exemplo
        extractor = EmbeddingExtractor(
            embedding_type=embedding_type,
            n_components=10 if embedding_type == 'tfidf_svd' else None
        )
        
        # Gerar embeddings
        embeddings = extractor.fit_transform(example_texts)
        
        print(f"Forma dos embeddings: {embeddings.shape}")
        print(f"Primeiros 5 valores do primeiro texto:")
        print(embeddings[0][:5])
        
        # Calcular similaridade entre textos
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print("\nMatriz de similaridade entre os textos:")
        print(similarities.round(3))
        print("\n" + "-"*50) 
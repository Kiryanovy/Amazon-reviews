"""
Text feature extraction for Amazon Reviews sentiment analysis.
"""

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

class TextFeatureExtractor:
    """Extracts text-based features from reviews."""
    
    def __init__(self, max_features=5000, n_components=100):
        self.max_features = max_features
        self.n_components = n_components
        
        # Inicializa os modelos com parâmetros ajustados para amostras pequenas
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=1,              # Aceita palavras que aparecem pelo menos 1 vez
            max_df=1.0,           # Aceita palavras que aparecem em até 100% dos documentos
            ngram_range=(1, 2)    # Considera unigramas e bigramas
        )
        
        # Usa TruncatedSVD para redução de dimensionalidade
        self.svd = TruncatedSVD(
            n_components=min(n_components, max_features),  # Garante que n_components não exceda max_features
            random_state=42
        )
    
    def extract_length_features(self, text):
        """Extract features related to text length."""
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'sentence_count': len(text.split('.')),
        }
    
    def extract_lexical_features(self, text):
        """Extract lexical features from text."""
        words = text.split()
        word_freq = Counter(words)
        
        return {
            'unique_words': len(set(words)),
            'unique_ratio': len(set(words)) / len(words) if words else 0,
            'hapax_ratio': len([w for w, c in word_freq.items() if c == 1]) / len(words) if words else 0
        }
    
    def extract_sentiment_features(self, text):
        """Extract basic sentiment-related features."""
        return {
            'has_exclamation': '!' in text,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capitalized_count': len(re.findall(r'\b[A-Z]+\b', text)),
            'positive_words': sum(1 for word in text.lower().split() if word in {'good', 'great', 'awesome', 'excellent', 'amazing', 'love', 'perfect', 'best'}),
            'negative_words': sum(1 for word in text.lower().split() if word in {'bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'horrible', 'disappointing'})
        }
    
    def extract_all_features(self, text):
        """Extract all available features from text."""
        features = {}
        features.update(self.extract_length_features(text))
        features.update(self.extract_lexical_features(text))
        features.update(self.extract_sentiment_features(text))
        return features
    
    def fit_transform(self, texts):
        """Fit and transform texts to feature matrix."""
        # Extrai features básicas
        basic_features = []
        for text in texts:
            features = self.extract_all_features(text)
            basic_features.append(list(features.values()))
        basic_features = np.array(basic_features)
        
        # Extrai features TF-IDF e aplica SVD
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Ajusta o número de componentes se necessário
        n_components = min(self.n_components, tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1)
        if n_components != self.svd.n_components:
            self.svd.n_components = n_components
        
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        
        # Combina todas as features
        return np.hstack([basic_features, tfidf_reduced])
    
    def transform(self, texts):
        """Transform texts to feature matrix."""
        # Extrai features básicas
        basic_features = []
        for text in texts:
            features = self.extract_all_features(text)
            basic_features.append(list(features.values()))
        basic_features = np.array(basic_features)
        
        # Extrai features TF-IDF e aplica SVD
        tfidf_matrix = self.tfidf.transform(texts)
        tfidf_reduced = self.svd.transform(tfidf_matrix)
        
        # Combina todas as features
        return np.hstack([basic_features, tfidf_reduced])

if __name__ == "__main__":
    # Exemplo de uso
    extractor = TextFeatureExtractor(max_features=100, n_components=10)  # Dimensões menores para exemplo
    
    # Textos de exemplo
    example_texts = [
        "This product is amazing! I love it so much. The quality is great.",
        "Terrible product. Don't waste your money. Poor quality and bad customer service!!!",
        "It's okay, nothing special. Average product for the price."
    ]
    
    print("Demonstração do TextFeatureExtractor:\n")
    
    # Extrair e mostrar features básicas para cada texto
    for i, text in enumerate(example_texts, 1):
        print(f"Texto {i}:", text)
        
        # Extrair cada tipo de feature separadamente
        print("\nCaracterísticas de comprimento:")
        length_features = extractor.extract_length_features(text)
        for feature, value in length_features.items():
            print(f"{feature}: {value:.2f}")
            
        print("\nCaracterísticas léxicas:")
        lexical_features = extractor.extract_lexical_features(text)
        for feature, value in lexical_features.items():
            print(f"{feature}: {value:.2f}")
            
        print("\nCaracterísticas de sentimento:")
        sentiment_features = extractor.extract_sentiment_features(text)
        for feature, value in sentiment_features.items():
            if isinstance(value, bool):
                print(f"{feature}: {value}")
            else:
                print(f"{feature}: {value}")
        
        print("\n" + "-"*50 + "\n")
    
    # Demonstrar transformação em matriz com TF-IDF e SVD
    feature_matrix = extractor.fit_transform(example_texts)
    print("Forma da matriz de features:", feature_matrix.shape)
    print("\nPrimeiros 5 valores do primeiro texto (combinação de features básicas e TF-IDF reduzido):")
    print(feature_matrix[0][:5])
    
    # Mostrar as palavras mais importantes do TF-IDF
    print("\nPalavras mais importantes (TF-IDF) para o primeiro texto:")
    feature_names = extractor.tfidf.get_feature_names_out()
    tfidf_matrix = extractor.tfidf.transform([example_texts[0]])
    nonzero_indices = tfidf_matrix.nonzero()[1]
    nonzero_values = tfidf_matrix.data
    sorted_indices = nonzero_values.argsort()[-5:][::-1]  # Top 5 palavras
    for idx in sorted_indices:
        word_idx = nonzero_indices[idx]
        print(f"{feature_names[word_idx]}: {nonzero_values[idx]:.4f}")
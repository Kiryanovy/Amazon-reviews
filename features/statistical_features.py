"""
Statistical feature extraction for Amazon Reviews sentiment analysis.
"""

import numpy as np
from collections import Counter
import re
from scipy import stats

class StatisticalFeatureExtractor:
    """Extracts statistical features from reviews."""
    
    def __init__(self):
        self.features = []
        self.word_stats = {}
    
    def extract_distribution_features(self, text):
        """Extract statistical distribution features."""
        words = text.split()
        word_lengths = [len(word) for word in words]
        
        if not word_lengths:
            return {
                'mean_length': 0,
                'std_length': 0,
                'skewness': 0,
                'kurtosis': 0
            }
        
        return {
            'mean_length': np.mean(word_lengths),
            'std_length': np.std(word_lengths),
            'skewness': stats.skew(word_lengths) if len(word_lengths) > 2 else 0,
            'kurtosis': stats.kurtosis(word_lengths) if len(word_lengths) > 2 else 0
        }
    
    def extract_frequency_features(self, text):
        """Extract word frequency based features."""
        words = text.lower().split()
        word_freq = Counter(words)
        
        if not words:
            return {
                'freq_ratio': 0,
                'rare_words_ratio': 0,
                'vocab_richness': 0
            }
        
        # Calculate frequency statistics
        freq_dist = list(word_freq.values())
        return {
            'freq_ratio': max(freq_dist) / min(freq_dist) if len(freq_dist) > 1 else 0,
            'rare_words_ratio': len([w for w, c in word_freq.items() if c == 1]) / len(words),
            'vocab_richness': len(set(words)) / np.sqrt(len(words))
        }
    
    def extract_pattern_features(self, text):
        """Extract pattern-based statistical features."""
        return {
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'space_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0
        }
    
    def extract_all_features(self, text):
        """Extract all statistical features."""
        features = {}
        features.update(self.extract_distribution_features(text))
        features.update(self.extract_frequency_features(text))
        features.update(self.extract_pattern_features(text))
        return features
    
    def fit_transform(self, texts):
        """Fit and transform texts to feature matrix."""
        feature_matrix = []
        for text in texts:
            features = self.extract_all_features(text)
            feature_matrix.append(list(features.values()))
        
        return np.array(feature_matrix)
    
    def transform(self, texts):
        """Transform texts to feature matrix."""
        return self.fit_transform(texts)

if __name__ == "__main__":
    # Exemplo de uso
    extractor = StatisticalFeatureExtractor()
    
    # Textos de exemplo
    example_texts = [
        "This product is amazing! I love it so much. The quality is great.",
        "Terrible product. Don't waste your money. Poor quality and bad customer service!!!",
        "It's okay, nothing special. Average product for the price."
    ]
    
    print("Demonstração do StatisticalFeatureExtractor:\n")
    
    # Extrair features para cada texto
    for i, text in enumerate(example_texts, 1):
        print(f"Texto {i}:", text)
        features = extractor.extract_all_features(text)
        print("\nCaracterísticas estatísticas:")
        for feature, value in features.items():
            print(f"{feature}: {value:.4f}")
        print("\n" + "-"*50 + "\n")
    
    # Demonstrar transformação em matriz
    feature_matrix = extractor.fit_transform(example_texts)
    print("Forma da matriz de features:", feature_matrix.shape)
    print("Primeiras 3 features da primeira review:", feature_matrix[0][:3]) 
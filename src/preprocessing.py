import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.data_loading import load_amazon_reviews
import os

def download_nltk_resources():
    """Download required NLTK resources"""
    # Define o diretório de dados do NLTK
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Lista de recursos necessários
    resources = [
        'stopwords',
        'wordnet',
        'omw-1.4'  # Open Multilingual Wordnet
    ]
    
    # Download de cada recurso
    for resource in resources:
        try:
            print(f"Baixando recurso {resource}...")
            nltk.download(resource, quiet=True, raise_on_error=True)
        except Exception as e:
            print(f"Erro ao baixar recurso {resource}: {e}")
            print("Tentando download alternativo...")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Falha no download alternativo de {resource}: {e}")
                return False
    return True

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, stemming=False, lemmatization=True):
        """
        Inicializa o preprocessador de texto
        Args:
            remove_stopwords: Se True, remove stopwords
            stemming: Se True, aplica stemming
            lemmatization: Se True, aplica lematização (ignorado se stemming=True)
        """
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization and not stemming
        
        # Inicializa recursos
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Aviso: Não foi possível carregar stopwords. Tentando download...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
        self.stemmer = PorterStemmer() if stemming else None
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None
    
    def clean_text(self, text):
        """Limpa e normaliza o texto"""
        if pd.isna(text):
            return ""
        
        # Converte para minúsculas
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove números
        text = re.sub(r'\d+', '', text)
        
        # Remove pontuação e caracteres especiais
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokeniza o texto em palavras usando uma abordagem simples
        Divide o texto em palavras usando espaços e remove tokens vazios
        """
        return [token.strip() for token in text.split() if token.strip()]
    
    def remove_stops(self, tokens):
        """Remove stopwords"""
        if self.remove_stopwords:
            return [t for t in tokens if t not in self.stop_words]
        return tokens
    
    def apply_stemming(self, tokens):
        """Aplica stemming nas palavras"""
        if self.stemming:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def apply_lemmatization(self, tokens):
        """Aplica lematização nas palavras"""
        if self.lemmatization:
            try:
                return [self.lemmatizer.lemmatize(t) for t in tokens]
            except:
                print("Aviso: Erro na lematização. Tentando download dos recursos...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens
    
    def preprocess(self, text):
        """Aplica todo o pipeline de pré-processamento"""
        # Limpeza básica
        text = self.clean_text(text)
        
        # Tokenização
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stops(tokens)
        
        # Stemming ou Lematização
        if self.stemming:
            tokens = self.apply_stemming(tokens)
        elif self.lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Remove tokens muito curtos (opcional)
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens

def preprocess_dataset(df, text_column='text', title_column='title', combine_title=True):
    """
    Pré-processa o dataset inteiro
    Args:
        df: DataFrame com os textos
        text_column: Nome da coluna com o texto principal
        title_column: Nome da coluna com o título
        combine_title: Se True, combina título e texto
    Returns:
        DataFrame com textos processados
    """
    # Inicializa o preprocessador
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        stemming=False,
        lemmatization=True
    )
    
    # Cria uma cópia do DataFrame
    processed_df = df.copy()
    
    # Combina título e texto se necessário
    if combine_title and title_column in df.columns:
        processed_df['full_text'] = df[title_column].astype(str) + " " + df[text_column].astype(str)
        text_to_process = 'full_text'
    else:
        text_to_process = text_column
    
    # Aplica o pré-processamento
    print("Iniciando pré-processamento dos textos...")
    processed_df['processed_tokens'] = processed_df[text_to_process].apply(preprocessor.preprocess)
    
    # Junta os tokens em um texto processado
    processed_df['processed_text'] = processed_df['processed_tokens'].apply(' '.join)
    
    return processed_df

if __name__ == "__main__":
    # Download recursos do NLTK
    print("Baixando recursos do NLTK...")
    if not download_nltk_resources():
        print("Erro ao baixar recursos do NLTK. O programa pode não funcionar corretamente.")
    
    # Carrega uma amostra do dataset
    print("Carregando dataset...")
    df = load_amazon_reviews(sample_size=1000)
    
    if df is not None:
        # Aplica pré-processamento
        processed_df = preprocess_dataset(df)
        
        # Mostra exemplos
        print("\nExemplos de textos processados:")
        for sentiment in [1, 2]:
            sample = processed_df[processed_df['sentiment'] == sentiment].iloc[0]
            print(f"\nSentimento: {'Negativo' if sentiment == 1 else 'Positivo'}")
            print(f"Texto original: {sample['text'][:200]}...")
            print(f"Texto processado: {sample['processed_text'][:200]}...")
            print(f"Tokens: {sample['processed_tokens'][:20]}...") 
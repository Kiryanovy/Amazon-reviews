import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import argparse

def load_local_csv(file_path, chunksize=None):
    """
    Load a local CSV file with optimizations for large files
    Args:
        file_path: Path to the CSV file
        chunksize: If provided, will load the file in chunks of this size
    """
    try:
        # Configurações específicas para leitura do CSV
        csv_params = {
            'names': ['sentiment', 'title', 'text'],  # Nomes das colunas
            'header': None,                  # Arquivo não tem cabeçalho
            'encoding': 'utf-8',            # Encoding do arquivo
            'quoting': 1,                   # Considerar aspas
            'on_bad_lines': 'skip',         # Pular linhas com erro
            'escapechar': '\\'              # Caractere de escape
        }
        
        if chunksize:
            csv_params['chunksize'] = chunksize
            
        # Carrega o arquivo
        df = pd.read_csv(file_path, **csv_params)
        
        if not chunksize:
            print(f"\nDataset {os.path.basename(file_path)} carregado!")
            print(f"Uso de memória: {df.memory_usage().sum() / 1024**2:.2f} MB")
            
            # Converte o sentimento para int depois de carregar
            df['sentiment'] = df['sentiment'].astype('int8')
            
        return df
            
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV {os.path.basename(file_path)}: {e}")
        return None

def load_amazon_reviews(source='local', train_path='train.csv', test_path='test.csv', use_chunks=False, chunksize=100000, sample_size=None):
    """
    Load Amazon reviews dataset
    Args:
        source: 'local' para arquivo local ou 'kaggle' para download
        train_path: Caminho para o arquivo CSV de treino
        test_path: Caminho para o arquivo CSV de teste
        use_chunks: Se True, carrega o arquivo em chunks
        chunksize: Tamanho de cada chunk
        sample_size: Se definido, carrega apenas uma amostra dos dados
    """
    if source == 'local':
        # Verifica se os arquivos existem
        if not os.path.exists(train_path):
            print(f"Arquivo de treino não encontrado: {train_path}")
            return None
        if not os.path.exists(test_path):
            print(f"Arquivo de teste não encontrado: {test_path}")
            return None
            
        # Carrega os datasets
        train_df = load_local_csv(train_path, chunksize if use_chunks else None)
        test_df = load_local_csv(test_path, chunksize if use_chunks else None)
        
        if use_chunks:
            return train_df, test_df
        
        if train_df is not None and test_df is not None:
            # Se sample_size for definido, pega uma amostra aleatória
            if sample_size:
                train_size = int(sample_size * 0.9)  # 90% para treino
                test_size = sample_size - train_size  # 10% para teste
                train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=42)
                test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=42)
            
            # Adiciona uma coluna para identificar a origem dos dados
            train_df['split'] = 'train'
            test_df['split'] = 'test'
            
            # Combina os datasets
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            print("\nDatasets combinados com sucesso!")
            print(f"Uso de memória total: {df.memory_usage().sum() / 1024**2:.2f} MB")
            return df
        return None
    
    # Mantém os métodos do Kaggle como fallback
    try:
        if source == 'kaggle_direct':
            path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
            if path:
                train_path = os.path.join(path, 'train.ft.txt.gz')
                test_path = os.path.join(path, 'test.ft.txt.gz')
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    train_df = pd.read_csv(train_path, compression='gzip', 
                                         header=None, names=['sentiment', 'text'],
                                         sep=' ', skiprows=0)
                    test_df = pd.read_csv(test_path, compression='gzip',
                                        header=None, names=['sentiment', 'text'],
                                        sep=' ', skiprows=0)
                    df = pd.concat([train_df, test_df], ignore_index=True)
                    df['sentiment'] = df['sentiment'].str.replace('__label__', '').astype(int)
                    return df
                else:
                    print(f"Dataset files not found in {path}")
                    return None
        else:
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "bittlingmayer/amazonreviews",
                "train.ft.txt.gz"
            )
            return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    # Configuração dos argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Carrega e processa o dataset de reviews da Amazon')
    parser.add_argument('--sample-size', type=int, help='Número de amostras para carregar (opcional)')
    args = parser.parse_args()
    
    # Carrega os datasets locais
    df = load_amazon_reviews(source='local', train_path='train.csv', test_path='test.csv', 
                           sample_size=args.sample_size)
    
    if df is not None:
        print("\nInformações do Dataset Completo:")
        print("Shape total:", df.shape)
        print("\nDistribuição train/test:")
        print(df['split'].value_counts())
        print("\nDistribuição de sentimentos:")
        sentiment_dist = df['sentiment'].value_counts()
        print("1 (Negativo):", sentiment_dist.get(1, 0))
        print("2 (Positivo):", sentiment_dist.get(2, 0))
        
        print("\nExemplos de Reviews:")
        print("\nReview Positiva:")
        positive = df[df['sentiment'] == 2].iloc[0]
        print(f"Título: {positive['title']}")
        print(f"Texto: {positive['text'][:200]}...")
        
        print("\nReview Negativa:")
        negative = df[df['sentiment'] == 1].iloc[0]
        print(f"Título: {negative['title']}")
        print(f"Texto: {negative['text'][:200]}...") 
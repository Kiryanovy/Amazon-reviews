from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carrega o modelo transformer
print("Carregando modelo de linguagem...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reviews para treinar o classificador
train_reviews = [
    # Reviews positivos com padrões similares
    ("Este produto é incrível!", "Positivo"),
    ("A qualidade é excelente!", "Positivo"),
    ("Preço muito justo pelo que oferece.", "Positivo"),
    ("Produto incrível, qualidade excepcional.", "Positivo"),
    ("Excelente qualidade e preço justo.", "Positivo"),
    ("Superou minhas expectativas!", "Positivo"),
    ("Recomendo fortemente!", "Positivo"),
    
    # Reviews positivos gerais
    ("Muito bom, adorei!", "Positivo"),
    ("Produto fantástico!", "Positivo"),
    ("Melhor compra que já fiz!", "Positivo"),
    ("Vale cada centavo!", "Positivo"),
    ("Ótimo custo-benefício.", "Positivo"),
    ("Qualidade impressionante.", "Positivo"),
    ("Estou muito satisfeito.", "Positivo"),
    
    # Reviews negativos
    ("Péssimo produto, não recomendo.", "Negativo"),
    ("Não gostei, qualidade ruim.", "Negativo"),
    ("Deixou a desejar.", "Negativo"),
    ("Péssimo custo-benefício.", "Negativo"),
    ("Produto com defeito.", "Negativo"),
    ("Qualidade deixa a desejar.", "Negativo"),
    ("Preço alto demais para o que oferece.", "Negativo"),
    ("Não vale o preço cobrado.", "Negativo"),
    ("Decepcionante.", "Negativo"),
    ("Esperava muito mais.", "Negativo")
]

# Reviews para testar
test_reviews = [
    {
        "texto": "Este produto é incrível! A qualidade é excelente e o preço é justo.",
        "tipo": "Positivo"
    },
    {
        "texto": "Não gostei do produto. Qualidade ruim e preço alto demais.",
        "tipo": "Negativo"
    },
    {
        "texto": "O produto chegou no prazo, mas a qualidade deixa a desejar.",
        "tipo": "Misto"
    },
    {
        "texto": "Comprei há 2 meses e já está com defeito. Péssimo custo-benefício.",
        "tipo": "Negativo"
    },
    {
        "texto": "Superou minhas expectativas! Recomendo fortemente.",
        "tipo": "Positivo"
    }
]

print("Preparando dados de treino...")
# Prepara os dados de treino
X_train = model.encode([review[0] for review in train_reviews])
y_train = [1 if review[1] == "Positivo" else 0 for review in train_reviews]

# Treina o classificador com parâmetros ajustados
print("Treinando classificador...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
classifier = LinearSVC(random_state=42, C=1.0, class_weight='balanced')
classifier.fit(X_train_scaled, y_train)

def classify_sentiment(text):
    # Gera embedding do texto
    embedding = model.encode([text])[0].reshape(1, -1)
    
    # Normaliza o embedding
    embedding_scaled = scaler.transform(embedding)
    
    # Faz a previsão
    prediction = classifier.predict(embedding_scaled)[0]
    
    return "Positivo" if prediction == 1 else "Negativo"

print("\n=== Teste de Classificação Automática de Reviews ===\n")

for review in test_reviews:
    print(f"\nTipo esperado: {review['tipo']}")
    print(f"Review: '{review['texto']}'")
    print(f"Classificação: {classify_sentiment(review['texto'])}")
    print("-" * 50) 
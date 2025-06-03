Feature: Análise de Sentimentos de Reviews
    Como um analista de dados
    Eu quero classificar automaticamente reviews de produtos
    Para entender o sentimento dos clientes

    Scenario: Classificar um review positivo
        Given um review com texto "Este produto é excelente, superou minhas expectativas!"
        When o modelo classifica o review
        Then o sentimento previsto deve ser "Positivo"

    Scenario: Classificar um review negativo
        Given um review com texto "Produto horrível, não recomendo para ninguém."
        When o modelo classifica o review
        Then o sentimento previsto deve ser "Negativo"

    Scenario: Pré-processar um texto de review
        Given um texto de review "Este PRODUTO é Muito BOM!!!"
        When o texto é pré-processado
        Then o resultado deve estar em minúsculas
        And não deve conter pontuação
        And deve estar tokenizado

    Scenario: Extrair features de um texto
        Given um texto pré-processado "produto excelente ótima qualidade"
        When as features são extraídas
        Then deve retornar um vetor de características
        And o vetor não deve conter valores nulos 
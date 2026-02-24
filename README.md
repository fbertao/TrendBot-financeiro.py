# TrendBot Financeiro

TrendBot financeiro surgiu da ideia inicial: preciso entender o funcionamento de machine learning, como parametrizar e como avaliar, como posso fazer isso envolvendo minha área de atuação? Foi então que comecei uma Iniciação Cientifícia pela Universidade Federal do ABC (UFABC), sob orientação do Prof. Mateus Coelho.

O foco principal é a previsão probabilística de movimentos direcionais (Up/Not Up) a partir de janelas deslizantes de preços históricos.
## Mas o que é o TrendBot?

O TrendBot é um sistema de apoio à decisão baseado em Deep Learning que analisa janelas deslizantes de 20 dias de preços históricos, gerando probabilidades de alta no próximo dia, assim permitindo ajustar o limitar (threshold) para o controle de risco.
O objetivo não é prever o mercado de forma determninística, mas sim gerar sinais probabilísticos que possam ser filtrados de acordo com o perfil de risco desejado.

# Pipeline do Projeto: 
## Coleta e Estrutura de dados: 
* Base: Kaggle - Stock Market Prediction
* Transformação das séries temporais em janelas deslizantes (20 dias)
* Conversão de preços em variações percentuais
* Remoção de outliers extremos
* Normalização via RobustScaler 

## Modelagem 
### Arquitetura utilizada:
* Dense (32) + ReLU
* Batch Normalization
* Dropout (0.2)
* Dense (16) + ReLU
* Batch Normalization
* Dropout (0.2)
* Dense (8) + ReLU
* Output Softmax (2 classes)

Técnincas aplicadas:
* Regularização para controle de overfitting
* Class weights para balanceamento
* Early stopping
* Learning rate reduction

## Avaliação do Modelo
### Métricas utilizadas
* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Matriz de confusão

Além disso, foi realizada análise com diferentes níveis de confiança (threshold), avaliando:
* Redução do número de operações
* Aumento da precisão
* Trade-off risco Vs Volume

# Próximos passos
- sistema de recomendações de ativos;
- com base no momento atual, qual ativo está bom para venda?

# Sistema de Análise de Concessão de Empréstimo

Este projeto é uma solução de Ciência de Dados para o problema de previsão de risco do cliente para concessão de empréstimos. Inclui um aplicativo web construído com Streamlit, uma análise em Jupyter Notebook, e um modelo preditivo treinado.

## Detalhes do Aplicativo
O aplicativo web permite inserir dados do cliente para análise de risco e realiza a predição usando um modelo treinado. Os atributos selecionados e parte dos dados são exibidos para análise.

## Estrutura do Projeto
```
sistema_classificacao_risco/
|-- dados/
|  -- risco.csv
|-- app_risco.py
|-- sistema_classificacao_risco.ipynb
|-- requirements.txt
|-- model_class_risco.joblib
|-- standard_scaler.joblib
|-- README.md
```

- **dados/risco.csv:** Arquivo contendo dados para treinar a máquina preditiva.
- **app_risco.py:** Código-fonte do aplicativo web.
- **sistema_classificacao_risco.ipynb:** Notebook com a análise, pré-processamento dos dados, treinamento e avaliação da máquina preditiva usando KNN Classifier.
- **requirements.txt:** Lista de bibliotecas necessárias.
- **model_class_risco.joblib:** Arquivo contendo o modelo preditivo treinado.
- **standard_scaler.joblib:** Arquivo com a padronização dos dados.

## Análise e Treinamento do Modelo
O notebook sistema_classificacao_risco.ipynb contém a análise exploratória de dados, pré-processamento, treinamento do modelo e avaliação.

## Executando o Aplicativo Web

Certifique-se de ter as bibliotecas necessárias instaladas:

```bash
pip install -r requirements.txt
```

Execute o aplicativo web:
```bash
streamlit run app_risco.py
```


# bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
import streamlit as st

st.set_page_config(
    page_title="Análise de concessão de empréstimo",
    page_icon="\U0001F4B2",  # Ícone opcional exibido na aba do navegador
    layout="wide",  # Padrão ou wide
    initial_sidebar_state="expanded",  # expanded ou collapsed
)

@st.cache_data
def get_data():
    return pd.read_csv("dados/risco.csv")


#obtem dados para treino do modelo preditivo
dados = get_data()

#carrega o modelo preditivo e a normalizacao feita no notebook
modelo = load("model_class_risco.joblib")
scaler = load("standard_scaler.joblib")

#aplicacao web


st.title("Sistema de previsão de risco para concessão de empréstimo")

st.markdown("Este é uma aplicativo utilizado para exibir a solução de Ciência de Dados para o problema de predição de Risco do cliente para concessão de empréstimos")


with st.sidebar:
    st.subheader("Insira os dados do cliente para análise de risco")
    indice_inad = st.number_input("Índice de inadimplência", value=dados['indice_inad'].mean()) #value é o valor inicial do campo
    anot_cadast = st.number_input("Anotações cadastrais (SPC / Serasa / ...)", value=dados['anot_cadastrais'].mean())
    class_renda = st.number_input("Classificação de renda", value=dados['class_renda'].mean())
    saldo_conta = st.number_input("Saldo de contas",value=dados['saldo_contas'].mean())
    btn_predicao = st.button("Realizar predição de risco")
     
if btn_predicao:
    dic_dados = {
        "anot_cadastrais" : [anot_cadast],
        "indice_inad" : [indice_inad],
        "class_renda" : [class_renda],
        "saldo_contas" : [saldo_conta]
    }
    
    df = pd.DataFrame(dic_dados)
    df = scaler.transform(df)
    
    resultado = modelo.predict(df)
    
    st.subheader("Concessão de empréstimo ao cliente")
    if(resultado == "Risco_Baixo"):
        st.success(f"**Classificação**: {resultado[0]}\n\n**Situação**: Aprovado")
    else:
        st.error(f"**Classificação**: {resultado[0]}\n\n**Situação**: Não Aprovado")
    


st.subheader("Selecionando as variáveis de análise dos clientes")

select_cols = st.multiselect(label="Atributos", options=dados.columns, 
                             default=["indice_inad", "anot_cadastrais", 
                                      "class_renda", "saldo_contas", "Risco"])

select_total = st.selectbox("Total de registros a serem exibidos", options=range(5, dados['id_cliente'].count()+1, 5), index=1)

if (select_cols or select_total):
    st.dataframe(dados[select_cols][0:select_total], hide_index=True)
else:
    st.error("Selecione ao menos um atributo!")
    

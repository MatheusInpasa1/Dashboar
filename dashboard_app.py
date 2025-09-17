# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os

def carregar_arquivo_excel():
    """
    Função para carregar um arquivo Excel adaptada para Streamlit
    """
    # Tentar primeiro o caminho original
    caminho_original = 'C:\\Users\\matheus.mendes\\Desktop\\Dashboard\\KPI - Utilidades 101623 rev5.xlsx'
    
    if os.path.exists(caminho_original):
        try:
            dados = pd.read_excel(caminho_original)
            st.success("Arquivo carregado com sucesso do caminho original!")
            return dados
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return None
    
    # Se não encontrar no caminho original, usar upload pelo Streamlit
    st.warning("Arquivo não encontrado no caminho original. Por favor, faça upload do arquivo.")
    
    uploaded_file = st.file_uploader(
        "Selecione o arquivo Excel 'KPI - Utilidades 101623 rev5.xlsx'",
        type=['xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            dados = pd.read_excel(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            return dados
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return None
    else:
        st.info("Aguardando upload do arquivo...")
        return None

def listar_arquivos_dashboard():
    """
    Função para listar arquivos no diretório Dashboard (para debugging)
    """
    diretorio = 'C:\\Users\\matheus.mendes\\Desktop\\Dashboard\\'
    if os.path.exists(diretorio):
        st.write("Arquivos no diretório Dashboard:")
        for arquivo in os.listdir(diretorio):
            if arquivo.endswith(('.xlsx', '.xls')) or 'KPI' in arquivo or 'Utilidades' in arquivo:
                st.write(f"  - {arquivo}")
    else:
        st.write(f"Diretório não encontrado: {diretorio}")

# Interface principal do Streamlit
def main():
    st.title("Dashboard de Utilidades")
    
    st.header("Carregamento de Dados")
    
    # Listar arquivos disponíveis (para debugging)
    with st.expander("Ver arquivos no diretório (debug)"):
        listar_arquivos_dashboard()
    
    # Carregar os dados
    dados = carregar_arquivo_excel()
    
    if dados is not None:
        st.success("Dados carregados com sucesso!")
        
        # Mostrar informações básicas dos dados
        st.subheader("Informações dos Dados")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", len(dados))
        
        with col2:
            st.metric("Total de Colunas", len(dados.columns))
        
        with col3:
            st.metric("Período dos Dados", f"{dados.iloc[:, 0].min()} a {dados.iloc[:, 0].max()}")
        
        # Mostrar preview dos dados
        st.subheader("Visualização dos Dados (Primeiras 10 linhas)")
        st.dataframe(dados.head(10))
        
        # Aqui você pode continuar com a parte de visualizações do seu dashboard
        # ... (seu código existente para gráficos e análises)
        
    else:
        st.error("Falha ao carregar os dados. Verifique o caminho do arquivo.")

if __name__ == "__main__":
    main()
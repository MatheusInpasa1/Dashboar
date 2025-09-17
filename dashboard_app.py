# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os

# Configuração da página
st.set_page_config(page_title="Dashboard de Utilidades", layout="wide")

# Função para carregar dados
@st.cache_data
def carregar_dados(uploaded_file):
    """Carrega os dados do arquivo Excel com cache para melhor performance"""
    try:
        dados = pd.read_excel(uploaded_file)
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def main():
    st.title("📊 Dashboard de Utilidades")
    
    # Sidebar para upload
    with st.sidebar:
        st.header("📁 Carregamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo Excel:",
            type=['xlsx', 'xls'],
            help="Faça upload do arquivo 'KPI - Utilidades 101623 rev5.xlsx'"
        )
        
        if uploaded_file is not None:
            st.success("✅ Arquivo selecionado!")
            st.write(f"**Nome:** {uploaded_file.name}")
            st.write(f"**Tamanho:** {uploaded_file.size / 1024:.1f} KB")
        else:
            st.info("📝 Aguardando upload do arquivo...")
            st.stop()  # Para a execução se não houver arquivo

    # Carregar dados
    dados = carregar_dados(uploaded_file)
    
    if dados is None:
        st.error("❌ Falha ao carregar os dados. Verifique o formato do arquivo.")
        st.stop()

    # ===== DADOS CARREGADOS COM SUCESSO =====
    st.success(f"✅ Dados carregados com sucesso! ({len(dados)} registros, {len(dados.columns)} colunas)")

    # Informações básicas
    st.header("📈 Informações dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(dados))
    with col2:
        st.metric("Total de Colunas", len(dados.columns))
    with col3:
        numeric_cols = len(dados.select_dtypes(include=[np.number]).columns)
        st.metric("Colunas Numéricas", numeric_cols)
    with col4:
        date_cols = len(dados.select_dtypes(include=['datetime64']).columns)
        st.metric("Colunas de Data", date_cols)

    # Visualização rápida dos dados
    with st.expander("👀 Visualizar Dados (Primeiras 10 linhas)"):
        st.dataframe(dados.head(10))

    # ===== ANÁLISES E VISUALIZAÇÕES =====
    st.header("📊 Análises e Visualizações")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal", "📋 Estatísticas", "🔥 Correlações", "🔍 Dispersão"])

    with tab1:
        st.subheader("Análise Temporal")
        
        # Encontrar coluna de data automaticamente
        date_cols = dados.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            coluna_data = date_cols[0]
            st.info(f"Coluna de data detectada: **{coluna_data}**")
        else:
            coluna_data = st.selectbox("Selecione a coluna de data:", options=dados.columns)
        
        coluna_metrica = st.selectbox("Selecione a métrica:", 
                                     options=[col for col in dados.columns if col != coluna_data])
        
        if coluna_data and coluna_metrica:
            fig = px.line(dados, x=coluna_data, y=coluna_metrica, 
                         title=f"Evolução de {coluna_metrica}",
                         labels={coluna_data: 'Data', coluna_metrica: 'Valor'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Estatísticas Descritivas")
        
        coluna_estatistica = st.selectbox("Selecione a coluna:", 
                                         options=dados.select_dtypes(include=[np.number]).columns)
        
        if coluna_estatistica:
            stats = dados[coluna_estatistica].describe()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Média", f"{stats['mean']:.2f}")
                st.metric("Mediana", f"{stats['50%']:.2f}")
            with col2:
                st.metric("Desvio Padrão", f"{stats['std']:.2f}")
                st.metric("Valor Máximo", f"{stats['max']:.2f}")
            
            fig = px.histogram(dados, x=coluna_estatistica, 
                              title=f"Distribuição de {coluna_estatistica}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Análise de Correlações")
        
        numeric_cols = dados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            if len(numeric_cols) > 10:
                selected_cols = st.multiselect("Selecione colunas para análise:", 
                                              options=numeric_cols, 
                                              default=numeric_cols[:5])
            else:
                selected_cols = numeric_cols
            
            if len(selected_cols) > 1:
                corr_matrix = dados[selected_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Matriz de Correlação",
                               color_continuous_scale="RdBu_r",
                               aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selecione pelo menos 2 colunas para análise de correlação")
        else:
            st.warning("Não há colunas numéricas suficientes para análise de correlação")

    with tab4:
        st.subheader("Gráfico de Dispersão")
        
        numeric_cols = dados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                coluna_x = st.selectbox("Eixo X:", options=numeric_cols)
            with col2:
                coluna_y = st.selectbox("Eixo Y:", options=numeric_cols)
            
            if coluna_x and coluna_y:
                fig = px.scatter(dados, x=coluna_x, y=coluna_y, 
                                title=f"{coluna_y} vs {coluna_x}",
                                trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não há colunas numéricas suficientes para gráfico de dispersão")

    # Download dos dados
    st.header("💾 Exportar Dados")
    csv = dados.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados como CSV",
        data=csv,
        file_name="dados_utilidades.csv",
        mime="text/csv",
        help="Clique para baixar os dados em formato CSV"
    )

if __name__ == "__main__":
    main()

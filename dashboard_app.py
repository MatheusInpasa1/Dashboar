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
    
    # Sidebar para upload e filtros
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
            st.stop()

    # Carregar dados
    dados = carregar_dados(uploaded_file)
    
    if dados is None:
        st.error("❌ Falha ao carregar os dados. Verifique o formato do arquivo.")
        st.stop()

    # ===== DADOS CARREGADOS COM SUCESSO =====
    st.success(f"✅ Dados carregados com sucesso! ({len(dados)} registros, {len(dados.columns)} colunas)")

    # ===== FILTROS GLOBAIS =====
    st.sidebar.header("🎛️ Filtros")
    
    # Filtro por colunas
    colunas_disponiveis = dados.columns.tolist()
    colunas_selecionadas = st.sidebar.multiselect(
        "Selecione as colunas para mostrar:",
        options=colunas_disponiveis,
        default=colunas_disponiveis[:min(10, len(colunas_disponiveis))]  # Mostra no máximo 10 colunas inicialmente
    )
    
    # Aplicar filtro de colunas
    if colunas_selecionadas:
        dados_filtrados = dados[colunas_selecionadas]
    else:
        dados_filtrados = dados

    # Informações básicas
    st.header("📈 Informações dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(dados_filtrados))
    with col2:
        st.metric("Total de Colunas", len(dados_filtrados.columns))
    with col3:
        numeric_cols = len(dados_filtrados.select_dtypes(include=[np.number]).columns)
        st.metric("Colunas Numéricas", numeric_cols)
    with col4:
        date_cols = len(dados_filtrados.select_dtypes(include=['datetime64']).columns)
        st.metric("Colunas de Data", date_cols)

    # Visualização rápida dos dados
    with st.expander("👀 Visualizar Dados Completos"):
        st.dataframe(dados_filtrados)

    # ===== ANÁLISES E VISUALIZAÇÕES =====
    st.header("📊 Análises e Visualizações")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal", "📋 Estatísticas", "🔥 Correlações", "🔍 Dispersão"])

    with tab1:
        st.subheader("Análise Temporal")
        
        # Encontrar coluna de data automaticamente
        date_cols = dados_filtrados.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            coluna_data = st.selectbox("Selecione a coluna de data:", 
                                      options=date_cols, 
                                      index=0)
        else:
            st.warning("⚠️ Nenhuma coluna de data encontrada")
            coluna_data = None
        
        if coluna_data:
            colunas_numericas = dados_filtrados.select_dtypes(include=[np.number]).columns.tolist()
            coluna_metrica = st.selectbox("Selecione a métrica:", options=colunas_numericas)
            
            if coluna_metrica:
                fig = px.line(dados_filtrados, x=coluna_data, y=coluna_metrica, 
                             title=f"Evolução de {coluna_metrica}",
                             labels={coluna_data: 'Data', coluna_metrica: 'Valor'})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Estatísticas Descritivas")
        
        colunas_numericas = dados_filtrados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_estatistica = st.selectbox("Selecione a coluna:", options=colunas_numericas)
            
            if coluna_estatistica:
                stats = dados_filtrados[coluna_estatistica].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Mediana", f"{stats['50%']:.2f}")
                with col3:
                    st.metric("Desvio Padrão", f"{stats['std']:.2f}")
                with col4:
                    st.metric("Valor Máximo", f"{stats['max']:.2f}")
                
                # Histograma
                fig = px.histogram(dados_filtrados, x=coluna_estatistica, 
                                  title=f"Distribuição de {coluna_estatistica}",
                                  nbins=30)
                st.plotly_chart(fig, use_container_width=True)
                
                # Boxplot
                fig2 = px.box(dados_filtrados, y=coluna_estatistica, 
                             title=f"Boxplot de {coluna_estatistica}")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Nenhuma coluna numérica encontrada para análise")

    with tab3:
        st.subheader("Análise de Correlações")
        
        numeric_cols = dados_filtrados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect(
                "Selecione colunas para análise de correlação:", 
                options=numeric_cols, 
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_cols) > 1:
                corr_matrix = dados_filtrados[selected_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Matriz de Correlação",
                               color_continuous_scale="RdBu_r",
                               aspect="auto",
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selecione pelo menos 2 colunas para análise de correlação")
        else:
            st.warning("Não há colunas numéricas suficientes para análise de correlação")

    with tab4:
        st.subheader("Gráfico de Dispersão")
        
        numeric_cols = dados_filtrados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                coluna_x = st.selectbox("Eixo X:", options=numeric_cols, key="scatter_x")
            with col2:
                # Garantir que o eixo Y seja diferente do X
                opcoes_y = [col for col in numeric_cols if col != coluna_x]
                if opcoes_y:
                    coluna_y = st.selectbox("Eixo Y:", options=opcoes_y, key="scatter_y")
                else:
                    coluna_y = None
                    st.warning("Não há outras colunas numéricas para o eixo Y")
            
            if coluna_x and coluna_y:
                # CORREÇÃO DO ERRO: Removido trendline="ols" que estava causando o problema
                fig = px.scatter(dados_filtrados, x=coluna_x, y=coluna_y, 
                                title=f"{coluna_y} vs {coluna_x}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não há colunas numéricas suficientes para gráfico de dispersão")

    # Download dos dados
    st.header("💾 Exportar Dados")
    csv = dados_filtrados.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados como CSV",
        data=csv,
        file_name="dados_utilidades_filtrados.csv",
        mime="text/csv",
        help="Clique para baixar os dados em formato CSV"
    )

if __name__ == "__main__":
    main()

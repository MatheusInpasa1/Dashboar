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

    # ===== FILTROS DE LINHAS =====
    st.sidebar.header("🎛️ Filtros de Linhas")
    
    # Filtro por colunas para seleção (apenas para visualização)
    colunas_disponiveis = dados.columns.tolist()
    colunas_selecionadas = st.sidebar.multiselect(
        "Colunas para visualização:",
        options=colunas_disponiveis,
        default=colunas_disponiveis[:min(8, len(colunas_disponiveis))]
    )
    
    # Filtros dinâmicos por coluna (para filtrar LINHAS)
    st.sidebar.header("🔍 Filtrar Linhas por Valores")
    
    filtros_aplicados = {}
    colunas_para_filtrar = st.sidebar.multiselect(
        "Selecione colunas para filtrar:",
        options=colunas_disponiveis,
        help="Escolha as colunas que deseja usar como filtro"
    )
    
    for coluna in colunas_para_filtrar:
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            # Filtro para colunas numéricas (slider)
            min_val = float(dados[coluna].min())
            max_val = float(dados[coluna].max())
            selected_range = st.sidebar.slider(
                f"Intervalo de {coluna}:",
                min_val, max_val, (min_val, max_val)
            )
            filtros_aplicados[coluna] = selected_range
        else:
            # Filtro para colunas categóricas (multiselect)
            unique_vals = dados[coluna].dropna().unique()
            selected_vals = st.sidebar.multiselect(
                f"Valores de {coluna}:",
                options=unique_vals,
                default=unique_vals[:min(5, len(unique_vals))]
            )
            filtros_aplicados[coluna] = selected_vals

    # Aplicar filtros às linhas
    dados_filtrados = dados.copy()
    
    for coluna, filtro in filtros_aplicados.items():
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            # Filtro numérico (intervalo)
            min_val, max_val = filtro
            dados_filtrados = dados_filtrados[
                (dados_filtrados[coluna] >= min_val) & 
                (dados_filtrados[coluna] <= max_val)
            ]
        else:
            # Filtro categórico (valores específicos)
            if filtro:  # Só filtra se algum valor foi selecionado
                dados_filtrados = dados_filtrados[dados_filtrados[coluna].isin(filtro)]
    
    # Aplicar filtro de colunas para visualização
    if colunas_selecionadas:
        dados_visualizacao = dados_filtrados[colunas_selecionadas]
    else:
        dados_visualizacao = dados_filtrados

    # Informações básicas
    st.header("📈 Informações dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(dados))
        st.caption("(Original)")
    with col2:
        st.metric("Registros Filtrados", len(dados_filtrados))
        st.caption("(Após filtros)")
    with col3:
        numeric_cols = len(dados_filtrados.select_dtypes(include=[np.number]).columns)
        st.metric("Colunas Numéricas", numeric_cols)
    with col4:
        date_cols = len(dados_filtrados.select_dtypes(include=['datetime64']).columns)
        st.metric("Colunas de Data", date_cols)

    # Visualização rápida dos dados
    with st.expander("👀 Visualizar Dados Filtrados"):
        st.dataframe(dados_visualizacao)

    # ===== SELETORES PARA GRÁFICOS =====
    st.header("🎯 Seleção de Variáveis para Gráficos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variáveis Disponíveis")
        todas_colunas = dados_filtrados.columns.tolist()
        colunas_numericas = dados_filtrados.select_dtypes(include=[np.number]).columns.tolist()
        colunas_categoricas = dados_filtrados.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_data = dados_filtrados.select_dtypes(include=['datetime64']).columns.tolist()
        
        st.write(f"**Numéricas:** {len(colunas_numericas)} colunas")
        st.write(f"**Categóricas:** {len(colunas_categoricas)} colunas")
        st.write(f"**Datas:** {len(colunas_data)} colunas")
    
    with col2:
        st.subheader("Configuração dos Gráficos")
        tipo_grafico = st.selectbox(
            "Tipo de Gráfico:",
            ["Linha", "Barras", "Dispersão", "Histograma", "Boxplot"]
        )

    # ===== ANÁLISES E VISUALIZAÇÕES =====
    st.header("📊 Análises e Visualizações")
    
    # Seletores específicos para cada tipo de gráfico
    if tipo_grafico == "Linha":
        st.subheader("📈 Gráfico de Linha")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eixo_x = st.selectbox("Eixo X (Data):", options=colunas_data + colunas_categoricas)
        with col2:
            eixo_y = st.selectbox("Eixo Y (Valor):", options=colunas_numericas)
        with col3:
            if colunas_categoricas:
                cor = st.selectbox("Cor (Agrupamento):", options=[None] + colunas_categoricas)
            else:
                cor = None
        
        if eixo_x and eixo_y:
            fig = px.line(dados_filtrados, x=eixo_x, y=eixo_y, color=cor,
                         title=f"{eixo_y} por {eixo_x}")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Barras":
        st.subheader("📊 Gráfico de Barras")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eixo_x = st.selectbox("Eixo X (Categoria):", options=colunas_categoricas + colunas_data)
        with col2:
            eixo_y = st.selectbox("Eixo Y (Valor):", options=colunas_numericas)
        with col3:
            if colunas_categoricas:
                cor = st.selectbox("Cor (Agrupamento):", options=[None] + colunas_categoricas)
            else:
                cor = None
        
        if eixo_x and eixo_y:
            fig = px.bar(dados_filtrados, x=eixo_x, y=eixo_y, color=cor,
                        title=f"{eixo_y} por {eixo_x}")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Dispersão":
        st.subheader("🔍 Gráfico de Dispersão")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eixo_x = st.selectbox("Eixo X:", options=colunas_numericas)
        with col2:
            eixo_y = st.selectbox("Eixo Y:", options=colunas_numericas)
        with col3:
            if colunas_categoricas:
                cor = st.selectbox("Cor (Agrupamento):", options=[None] + colunas_categoricas)
            else:
                cor = None
        
        if eixo_x and eixo_y:
            fig = px.scatter(dados_filtrados, x=eixo_x, y=eixo_y, color=cor,
                            title=f"{eixo_y} vs {eixo_x}")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Histograma":
        st.subheader("📋 Histograma")
        
        col1, col2 = st.columns(2)
        with col1:
            variavel = st.selectbox("Variável:", options=colunas_numericas)
        with col2:
            if colunas_categoricas:
                cor = st.selectbox("Cor (Agrupamento):", options=[None] + colunas_categoricas)
            else:
                cor = None
        
        if variavel:
            fig = px.histogram(dados_filtrados, x=variavel, color=cor,
                              title=f"Distribuição de {variavel}")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Boxplot":
        st.subheader("📦 Boxplot")
        
        col1, col2 = st.columns(2)
        with col1:
            variavel = st.selectbox("Variável:", options=colunas_numericas)
        with col2:
            if colunas_categoricas:
                eixo_x = st.selectbox("Agrupamento:", options=[None] + colunas_categoricas)
            else:
                eixo_x = None
        
        if variavel:
            fig = px.box(dados_filtrados, y=variavel, x=eixo_x,
                        title=f"Boxplot de {variavel}")
            st.plotly_chart(fig, use_container_width=True)

    # Download dos dados filtrados
    st.header("💾 Exportar Dados")
    csv = dados_filtrados.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados filtrados como CSV",
        data=csv,
        file_name="dados_utilidades_filtrados.csv",
        mime="text/csv",
        help="Clique para baixar os dados após aplicação dos filtros"
    )

if __name__ == "__main__":
    main()

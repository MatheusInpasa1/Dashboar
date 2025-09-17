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
            # Tenta encontrar uma coluna de data para mostrar o período
            date_columns = dados.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                min_date = dados[date_columns[0]].min()
                max_date = dados[date_columns[0]].max()
                st.metric("Período dos Dados", f"{min_date} a {max_date}")
            else:
                st.metric("Dados Carregados", "✓")
        
        # Mostrar preview dos dados
        st.subheader("Visualização dos Dados (Primeiras 10 linhas)")
        st.dataframe(dados.head(10))
        
        # =============================================
        # NOVAS VISUALIZAÇÕES ADICIONADAS AQUI
        # =============================================
        
        st.header("📊 Visualizações dos Dados")
        
        # 1. Seletor de tipo de análise
        tipo_analise = st.selectbox(
            "Selecione o tipo de análise:",
            ["Evolução Temporal", "Estatísticas", "Correlações", "Dispersão"]
        )
        
        if tipo_analise == "Evolução Temporal":
            st.subheader("📈 Evolução Temporal")
            
            # Encontrar coluna de data automaticamente
            date_cols = dados.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                coluna_data = date_cols[0]
            else:
                coluna_data = st.selectbox("Selecione a coluna de data:", options=dados.columns)
            
            coluna_metrica = st.selectbox("Selecione a métrica para análise:", 
                                         options=[col for col in dados.columns if col != coluna_data])
            
            if coluna_data and coluna_metrica:
                fig = px.line(dados, x=coluna_data, y=coluna_metrica, 
                             title=f"Evolução de {coluna_metrica} ao longo do tempo")
                st.plotly_chart(fig)
        
        elif tipo_analise == "Estatísticas":
            st.subheader("📋 Estatísticas Descritivas")
            
            coluna_estatistica = st.selectbox("Selecione a coluna para análise estatística:", 
                                             options=dados.select_dtypes(include=[np.number]).columns)
            
            if coluna_estatistica:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Média", f"{dados[coluna_estatistica].mean():.2f}")
                    st.metric("Mediana", f"{dados[coluna_estatistica].median():.2f}")
                
                with col2:
                    st.metric("Desvio Padrão", f"{dados[coluna_estatistica].std():.2f}")
                    st.metric("Valor Máximo", f"{dados[coluna_estatistica].max():.2f}")
                
                # Histograma
                fig = px.histogram(dados, x=coluna_estatistica, 
                                  title=f"Distribuição de {coluna_estatistica}")
                st.plotly_chart(fig)
        
        elif tipo_analise == "Correlações":
            st.subheader("🔥 Mapa de Calor de Correlações")
            
            # Seleciona apenas colunas numéricas
            numeric_cols = dados.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Limita para as primeiras 15 colunas para não ficar muito pesado
                if len(numeric_cols) > 15:
                    numeric_cols = numeric_cols[:15]
                    st.info("Mostrando apenas as primeiras 15 colunas numéricas")
                
                corr_matrix = dados[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Matriz de Correlação",
                               color_continuous_scale="RdBu_r",
                               aspect="auto")
                st.plotly_chart(fig)
            else:
                st.warning("Não há colunas numéricas suficientes para análise de correlação")
        
        elif tipo_analise == "Dispersão":
            st.subheader("🔍 Gráfico de Dispersão")
            
            numeric_cols = dados.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                coluna_x = st.selectbox("Eixo X:", options=numeric_cols)
                coluna_y = st.selectbox("Eixo Y:", options=numeric_cols)
                
                if coluna_x and coluna_y:
                    fig = px.scatter(dados, x=coluna_x, y=coluna_y, 
                                    title=f"{coluna_y} vs {coluna_x}",
                                    trendline="ols")
                    st.plotly_chart(fig)
            else:
                st.warning("Não há colunas numéricas suficientes para gráfico de dispersão")
        
        # Download dos dados processados
        st.subheader("💾 Download dos Dados")
        csv = dados.to_csv(index=False)
        st.download_button(
            label="Baixar dados como CSV",
            data=csv,
            file_name="dados_processados.csv",
            mime="text/csv"
        )
        
    else:
        st.error("Falha ao carregar os dados. Verifique o caminho do arquivo.")

if __name__ == "__main__":
    main()

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

# Função inteligente para converter para data
def converter_para_data(coluna):
    """Tenta converter uma coluna para datetime de várias formas"""
    try:
        # Tentativa 1: Converter diretamente
        return pd.to_datetime(coluna)
    except:
        try:
            # Tentativa 2: Converter com diferentes formatos
            return pd.to_datetime(coluna, dayfirst=True)
        except:
            try:
                # Tentativa 3: Converter como string e depois para data
                return pd.to_datetime(coluna.astype(str))
            except:
                # Se nada funcionar, retorna a coluna original
                return coluna

# Função para detectar automaticamente colunas de data
def detectar_colunas_data(dados):
    """Detecta automaticamente colunas que podem ser datas"""
    colunas_data = []
    
    for col in dados.columns:
        # Verifica se já é datetime
        if pd.api.types.is_datetime64_any_dtype(dados[col]):
            colunas_data.append(col)
        # Verifica se a coluna tem nome que sugere ser data
        elif any(palavra in col.lower() for palavra in ['data', 'date', 'dia', 'time', 'hora', 'timestamp']):
            colunas_data.append(col)
        # Verifica se pode ser convertido para datetime
        elif dados[col].dtype == 'object':
            try:
                # Testa com uma amostra
                amostra = dados[col].head(10).dropna()
                if len(amostra) > 0:
                    pd.to_datetime(amostra, errors='coerce')
                    colunas_data.append(col)
            except:
                pass
    
    return colunas_data

def main():
    st.title("📊 Dashboard de Utilidades - Análise Completa")
    
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
            st.stop()

    # Carregar dados
    dados = carregar_dados(uploaded_file)
    
    if dados is None:
        st.error("❌ Falha ao carregar os dados. Verifique o formato do arquivo.")
        st.stop()

    # ===== DADOS CARREGADOS COM SUCESSO =====
    st.success(f"✅ Dados carregados com sucesso! ({len(dados)} registros, {len(dados.columns)} colunas)")

    # ===== DETECÇÃO AUTOMÁTICA DE COLUNAS =====
    colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
    colunas_data = detectar_colunas_data(dados)
    
    # Mostrar informações das colunas detectadas
    st.sidebar.header("🔍 Colunas Detectadas")
    st.sidebar.write(f"**Numéricas:** {len(colunas_numericas)}")
    st.sidebar.write(f"**Possíveis Datas:** {len(colunas_data)}")
    
    if colunas_data:
        st.sidebar.write("Colunas detectadas como datas:")
        for col in colunas_data:
            st.sidebar.write(f"• {col}")

    # ===== INFORMACOES BASICAS =====
    st.header("📈 Informações dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(dados))
    with col2:
        st.metric("Total de Colunas", len(dados.columns))
    with col3:
        st.metric("Colunas Numéricas", len(colunas_numericas))
    with col4:
        st.metric("Colunas de Data", len(colunas_data))

    # Visualização rápida dos dados
    with st.expander("👀 Visualizar Dados Completos"):
        st.dataframe(dados)

    # ===== ANALISES ESTATISTICAS COMPLETAS =====
    st.header("📊 Análises Estatísticas Completas")
    
    # Abas para diferentes tipos de análise
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Séries Temporais", "📋 Estatísticas", "🔥 Correlações", 
        "📊 Distribuições", "🔍 Dispersão", "📦 Boxplots"
    ])

    with tab1:
        st.subheader("📈 Análise de Séries Temporais")
        
        if colunas_data and colunas_numericas:
            col1, col2 = st.columns(2)
            with col1:
                coluna_data = st.selectbox("Coluna de Data:", colunas_data, key="temp_data")
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas, key="temp_valor")
            
            if coluna_data and coluna_valor:
                # Tentar converter a coluna de data
                dados_temp = dados.copy()
                dados_temp['Data_Convertida'] = converter_para_data(dados_temp[coluna_data])
                
                # Verificar se a conversão foi bem-sucedida
                if pd.api.types.is_datetime64_any_dtype(dados_temp['Data_Convertida']):
                    # Ordenar por data
                    dados_temp = dados_temp.sort_values(by='Data_Convertida')
                    
                    fig = px.line(dados_temp, x='Data_Convertida', y=coluna_valor, 
                                 title=f"Evolução Temporal de {coluna_valor}",
                                 labels={'Data_Convertida': 'Data', coluna_valor: 'Valor'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estatísticas temporais
                    st.subheader("📋 Estatísticas Temporais")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Primeira Data", dados_temp['Data_Convertida'].min().strftime('%d/%m/%Y'))
                    with col2:
                        st.metric("Última Data", dados_temp['Data_Convertida'].max().strftime('%d/%m/%Y'))
                    with col3:
                        st.metric("Média", f"{dados_temp[coluna_valor].mean():.2f}")
                    with col4:
                        if len(dados_temp) > 1:
                            crescimento = ((dados_temp[coluna_valor].iloc[-1] - dados_temp[coluna_valor].iloc[0]) / dados_temp[coluna_valor].iloc[0] * 100) if dados_temp[coluna_valor].iloc[0] != 0 else 0
                            st.metric("Crescimento", f"{crescimento:.1f}%")
                        else:
                            st.metric("Crescimento", "N/A")
                else:
                    st.warning(f"⚠️ Não foi possível converter a coluna '{coluna_data}' para data.")
                    st.info("💡 Dica: Verifique o formato das datas no arquivo Excel.")
                    
                    # Mostrar amostra dos dados da coluna de data
                    st.write("**Amostra dos dados da coluna de data:**")
                    st.write(dados_temp[coluna_data].head(10).values)
        else:
            if not colunas_data:
                st.warning("❌ Nenhuma coluna de data detectada. Verifique se existe uma coluna com datas.")
            if not colunas_numericas:
                st.warning("❌ Nenhuma coluna numérica detectada.")

    with tab2:
        st.subheader("📋 Estatísticas Descritivas por Coluna")
        
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Selecione a coluna:", colunas_numericas, key="stats_col")
            
            if coluna_selecionada:
                stats = dados[coluna_selecionada].describe()
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Mediana", f"{stats['50%']:.2f}")
                with col3:
                    st.metric("Desvio Padrão", f"{stats['std']:.2f}")
                with col4:
                    cv = (stats['std']/stats['mean'])*100 if stats['mean'] != 0 else 0
                    st.metric("Coef. Variação", f"{cv:.1f}%")
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab3:
        st.subheader("🔥 Análise de Correlações")
        
        if len(colunas_numericas) > 1:
            # Matriz de correlação completa
            corr_matrix = dados[colunas_numericas].corr()
            
            fig = px.imshow(corr_matrix, 
                           title="Matriz de Correlação",
                           color_continuous_scale="RdBu_r",
                           aspect="auto",
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("❌ Número insuficiente de colunas numéricas")

    with tab4:
        st.subheader("📊 Análise de Distribuições")
        
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para análise de distribuição:", colunas_numericas, key="dist_col")
            
            if coluna_selecionada:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma
                    fig_hist = px.histogram(dados, x=coluna_selecionada, 
                                          title=f"Distribuição de {coluna_selecionada}",
                                          nbins=30)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Boxplot
                    fig_box = px.box(dados, y=coluna_selecionada, 
                                   title=f"Boxplot de {coluna_selecionada}")
                    st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab5:
        st.subheader("🔍 Gráficos de Dispersão")
        
        if len(colunas_numericas) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X:", colunas_numericas, key="scatter_x")
            with col2:
                eixo_y = st.selectbox("Eixo Y:", colunas_numericas, key="scatter_y")
            
            if eixo_x and eixo_y:
                fig = px.scatter(dados, x=eixo_x, y=eixo_y, 
                                title=f"{eixo_y} vs {eixo_x}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("❌ Número insuficiente de colunas numéricas")

    with tab6:
        st.subheader("📦 Boxplots por Variável")
        
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para boxplot:", colunas_numericas, key="boxplot_col")
            
            if coluna_selecionada:
                fig = px.box(dados, y=coluna_selecionada, 
                            title=f"Boxplot de {coluna_selecionada}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    # ===== DOWNLOAD DOS DADOS =====
    st.header("💾 Exportar Resultados")
    
    csv = dados.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados completos (CSV)",
        data=csv,
        file_name="dados_utilidades_completos.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

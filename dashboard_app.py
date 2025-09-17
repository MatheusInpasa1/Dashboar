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
        return pd.to_datetime(coluna, dayfirst=True, errors='coerce')
    except:
        return coluna

def main():
    st.title("📊 Dashboard de Utilidades - Análise Estatística")
    
    # Sidebar para upload
    with st.sidebar:
        st.header("📁 Carregamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo Excel:",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            st.success("✅ Arquivo selecionado!")
        else:
            st.info("📝 Aguardando upload do arquivo...")
            st.stop()

    # Carregar dados
    dados = carregar_dados(uploaded_file)
    
    if dados is None:
        st.error("❌ Falha ao carregar os dados.")
        st.stop()

    # ===== DADOS CARREGADOS COM SUCESSO =====
    st.success(f"✅ Dados carregados com sucesso! ({len(dados)} registros, {len(dados.columns)} colunas)")

    # ===== DETECÇÃO AUTOMÁTICA DE COLUNAS =====
    colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
    
    # Detectar colunas de data
    colunas_data = []
    for col in dados.columns:
        if any(palavra in col.lower() for palavra in ['data', 'date', 'dia', 'time']):
            colunas_data.append(col)

    # ===== ANÁLISE DE SÉRIES TEMPORAIS =====
    if colunas_data and colunas_numericas:
        st.header("📈 Análise de Séries Temporais")
        
        col1, col2 = st.columns(2)
        with col1:
            coluna_data = st.selectbox("Coluna de Data:", colunas_data)
        with col2:
            coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas)
        
        if coluna_data and coluna_valor:
            # Converter data
            dados_temp = dados.copy()
            dados_temp['Data_Convertida'] = converter_para_data(dados_temp[coluna_data])
            
            if pd.api.types.is_datetime64_any_dtype(dados_temp['Data_Convertida']):
                dados_temp = dados_temp.sort_values(by='Data_Convertida')
                
                # Gráfico de linha
                fig = px.line(dados_temp, x='Data_Convertida', y=coluna_valor, 
                             title=f"Evolução Temporal de {coluna_valor}")
                st.plotly_chart(fig, use_container_width=True)
                
                # ===== ESTATÍSTICAS DETALHADAS =====
                st.subheader("📊 Estatísticas Detalhadas")
                
                # Métricas básicas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{dados_temp[coluna_valor].mean():.2f}")
                with col2:
                    st.metric("Mediana", f"{dados_temp[coluna_valor].median():.2f}")
                with col3:
                    st.metric("Moda", f"{dados_temp[coluna_valor].mode().iloc[0] if not dados_temp[coluna_valor].mode().empty else 'N/A'}")
                with col4:
                    st.metric("Desvio Padrão", f"{dados_temp[coluna_valor].std():.2f}")
                
                # Quartis e intervalos
                Q1 = dados_temp[coluna_valor].quantile(0.25)
                Q3 = dados_temp[coluna_valor].quantile(0.75)
                IQR = Q3 - Q1
                
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Q1 (25%)", f"{Q1:.2f}")
                with col6:
                    st.metric("Q3 (75%)", f"{Q3:.2f}")
                with col7:
                    st.metric("IQR", f"{IQR:.2f}")
                with col8:
                    st.metric("Amplitude", f"{dados_temp[coluna_valor].max() - dados_temp[coluna_valor].min():.2f}")
                
                # Tendência e crescimento
                if len(dados_temp) > 1:
                    crescimento = ((dados_temp[coluna_valor].iloc[-1] - dados_temp[coluna_valor].iloc[0]) / dados_temp[coluna_valor].iloc[0] * 100) if dados_temp[coluna_valor].iloc[0] != 0 else 0
                    
                    col9, col10 = st.columns(2)
                    with col9:
                        st.metric("Crescimento Total", f"{crescimento:.1f}%")
                    with col10:
                        # Tendência linear simples
                        x = np.arange(len(dados_temp))
                        y = dados_temp[coluna_valor].values
                        coef = np.polyfit(x, y, 1)[0]
                        tendencia = "↗️ Alta" if coef > 0 else "↘️ Baixa" if coef < 0 else "➡️ Estável"
                        st.metric("Tendência", tendencia)
                
                # Informações temporais
                st.subheader("📅 Informações Temporais")
                col11, col12, col13, col14 = st.columns(4)
                with col11:
                    st.metric("Primeira Data", dados_temp['Data_Convertida'].min().strftime('%d/%m/%Y'))
                with col12:
                    st.metric("Última Data", dados_temp['Data_Convertida'].max().strftime('%d/%m/%Y'))
                with col13:
                    st.metric("Período", f"{(dados_temp['Data_Convertida'].max() - dados_temp['Data_Convertida'].min()).days} dias")
                with col14:
                    st.metric("Dados por Dia", f"{len(dados_temp)/max(1, (dados_temp['Data_Convertida'].max() - dados_temp['Data_Convertida'].min()).days):.1f}")
            
            else:
                st.warning("Não foi possível converter as datas. Mostrando análise sem ordenação temporal.")
                fig = px.line(dados_temp, x=coluna_data, y=coluna_valor, 
                             title=f"Evolução de {coluna_valor}")
                st.plotly_chart(fig, use_container_width=True)

    # ===== ANÁLISE ESTATÍSTICA GERAL =====
    st.header("📋 Análise Estatística Geral")
    
    if colunas_numericas:
        coluna_analise = st.selectbox("Selecione a coluna para análise detalhada:", colunas_numericas)
        
        if coluna_analise:
            # Estatísticas descritivas
            stats = dados[coluna_analise].describe()
            
            st.subheader("Estatísticas Descritivas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média", f"{stats['mean']:.2f}")
                st.metric("Moda", f"{dados[coluna_analise].mode().iloc[0] if not dados[coluna_analise].mode().empty else 'N/A'}")
            with col2:
                st.metric("Mediana", f"{stats['50%']:.2f}")
                st.metric("Desvio Padrão", f"{stats['std']:.2f}")
            with col3:
                st.metric("Mínimo", f"{stats['min']:.2f}")
                st.metric("Máximo", f"{stats['max']:.2f}")
            with col4:
                st.metric("Q1 (25%)", f"{stats['25%']:.2f}")
                st.metric("Q3 (75%)", f"{stats['75%']:.2f}")
            
            # Visualizações
            st.subheader("Visualizações")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(dados, x=coluna_analise, 
                                      title=f"Distribuição de {coluna_analise}",
                                      nbins=30)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(dados, y=coluna_analise, 
                               title=f"Boxplot de {coluna_analise}")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Análise de outliers
            Q1 = dados[coluna_analise].quantile(0.25)
            Q3 = dados[coluna_analise].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = dados[(dados[coluna_analise] < lower_bound) | (dados[coluna_analise] > upper_bound)]
            
            st.metric("Número de Outliers", len(outliers))
            
            if len(outliers) > 0:
                with st.expander("Ver Outliers"):
                    st.dataframe(outliers[[coluna_analise]])

    # ===== ANÁLISE DE CORRELAÇÕES =====
    if len(colunas_numericas) > 1:
        st.header("🔥 Análise de Correlações")
        
        corr_matrix = dados[colunas_numericas].corr()
        fig = px.imshow(corr_matrix, 
                       title="Matriz de Correlação",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlações
        st.subheader("Correlações mais Fortes")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'Variável 1': corr_matrix.columns[i],
                    'Variável 2': corr_matrix.columns[j],
                    'Correlação': corr_matrix.iloc[i, j]
                })
        
        df_corr = pd.DataFrame(correlations)
        df_corr['Abs_Correlation'] = df_corr['Correlação'].abs()
        top_correlations = df_corr.nlargest(5, 'Abs_Correlation')
        
        for _, row in top_correlations.iterrows():
            st.write(f"**{row['Variável 1']}** ↔ **{row['Variável 2']}**: {row['Correlação']:.3f}")

    # ===== GRÁFICOS DE DISPERSÃO =====
    if len(colunas_numericas) >= 2:
        st.header("🔍 Gráficos de Dispersão")
        
        col1, col2 = st.columns(2)
        with col1:
            eixo_x = st.selectbox("Eixo X:", colunas_numericas)
        with col2:
            eixo_y = st.selectbox("Eixo Y:", colunas_numericas)
        
        if eixo_x and eixo_y:
            fig = px.scatter(dados, x=eixo_x, y=eixo_y, 
                            title=f"{eixo_y} vs {eixo_x}")
            st.plotly_chart(fig, use_container_width=True)
            
            correlacao = dados[eixo_x].corr(dados[eixo_y])
            st.metric("Coeficiente de Correlação", f"{correlacao:.3f}")

    # ===== DOWNLOAD DOS DADOS =====
    st.header("💾 Exportar Resultados")
    
    csv = dados.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados completos (CSV)",
        data=csv,
        file_name="dados_analise.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

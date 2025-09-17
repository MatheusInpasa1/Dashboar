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

# Função para calcular intervalo de confiança sem scipy
def calcular_intervalo_confianca(data, confidence=0.95):
    """Calcula intervalo de confiança usando apenas numpy/pandas"""
    n = len(data)
    m = data.mean()
    se = data.std() / np.sqrt(n)
    # Usando aproximação normal para grandes amostras
    h = 1.96 * se  # 1.96 para 95% de confiança
    return m - h, m + h

# Função para teste t simplificado
def teste_t_simplificado(data):
    """Teste t simplificado sem scipy"""
    t_stat = data.mean() / (data.std() / np.sqrt(len(data)))
    # aproximação do valor-p para grandes amostras
    return t_stat

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

    # ===== INFORMACOES BASICAS =====
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
    with st.expander("👀 Visualizar Dados Completos"):
        st.dataframe(dados)

    # ===== ANALISES ESTATISTICAS COMPLETAS =====
    st.header("📊 Análises Estatísticas Completas")
    
    # Abas para diferentes tipos de análise
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Séries Temporais", "📋 Estatísticas", "🔥 Correlações", 
        "📊 Distribuições", "🔍 Dispersão", "📦 Boxplots", "🧮 Estatísticas Avançadas"
    ])

    with tab1:
        st.subheader("📈 Análise de Séries Temporais")
        
        colunas_data = dados.select_dtypes(include=['datetime64']).columns.tolist()
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        
        if colunas_data and colunas_numericas:
            col1, col2 = st.columns(2)
            with col1:
                coluna_data = st.selectbox("Coluna de Data:", colunas_data, key="temp_data")
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas, key="temp_valor")
            
            if coluna_data and coluna_valor:
                fig = px.line(dados, x=coluna_data, y=coluna_valor, 
                             title=f"Evolução Temporal de {coluna_valor}",
                             labels={coluna_data: 'Data', coluna_valor: 'Valor'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas temporais
                st.subheader("📋 Estatísticas Temporais")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Primeira Data", dados[coluna_data].min().date())
                with col2:
                    st.metric("Última Data", dados[coluna_data].max().date())
                with col3:
                    st.metric("Média", f"{dados[coluna_valor].mean():.2f}")
                with col4:
                    crescimento = ((dados[coluna_valor].iloc[-1] - dados[coluna_valor].iloc[0]) / dados[coluna_valor].iloc[0] * 100) if dados[coluna_valor].iloc[0] != 0 else 0
                    st.metric("Crescimento", f"{crescimento:.1f}%")
        else:
            st.warning("❌ Dados insuficientes para análise temporal")

    with tab2:
        st.subheader("📋 Estatísticas Descritivas por Coluna")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
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
                
                # Métricas secundárias
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Mínimo", f"{stats['min']:.2f}")
                with col6:
                    st.metric("Máximo", f"{stats['max']:.2f}")
                with col7:
                    st.metric("Q1 (25%)", f"{stats['25%']:.2f}")
                with col8:
                    st.metric("Q3 (75%)", f"{stats['75%']:.2f}")
                
                # Informações adicionais
                skewness = dados[coluna_selecionada].skew()
                kurtosis = dados[coluna_selecionada].kurtosis()
                
                col9, col10 = st.columns(2)
                with col9:
                    st.metric("Assimetria", f"{skewness:.2f}")
                with col10:
                    st.metric("Curtose", f"{kurtosis:.2f}")
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab3:
        st.subheader("🔥 Análise de Correlações")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(colunas_numericas) > 1:
            # Matriz de correlação completa
            corr_matrix = dados[colunas_numericas].corr()
            
            fig = px.imshow(corr_matrix, 
                           title="Matriz de Correlação",
                           color_continuous_scale="RdBu_r",
                           aspect="auto",
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlações
            st.subheader("🔝 Top Correlações")
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
            top_correlations = df_corr.nlargest(10, 'Abs_Correlation')
            
            st.dataframe(top_correlations[['Variável 1', 'Variável 2', 'Correlação']].style.format({
                'Correlação': '{:.3f}'
            }))
        else:
            st.warning("❌ Número insuficiente de colunas numéricas")

    with tab4:
        st.subheader("📊 Análise de Distribuições")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
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
                    # Gráfico de densidade
                    fig_density = px.histogram(dados, x=coluna_selecionada, 
                                             title=f"Densidade de {coluna_selecionada}",
                                             nbins=30, histnorm='probability density')
                    st.plotly_chart(fig_density, use_container_width=True)
                
                # Análise de normalidade simplificada
                st.subheader("📋 Análise de Normalidade")
                data_clean = dados[coluna_selecionada].dropna()
                
                col3, col4 = st.columns(2)
                with col3:
                    # Coeficiente de assimetria
                    skew = data_clean.skew()
                    st.metric("Coef. Assimetria", f"{skew:.3f}")
                    if abs(skew) < 0.5:
                        st.success("Distribuição aproximadamente simétrica")
                    elif abs(skew) < 1:
                        st.warning("Distribuição moderadamente assimétrica")
                    else:
                        st.error("Distribuição fortemente assimétrica")
                
                with col4:
                    # Coeficiente de curtose
                    kurt = data_clean.kurtosis()
                    st.metric("Coef. Curtose", f"{kurt:.3f}")
                    if abs(kurt) < 0.5:
                        st.success("Curtose próxima da normal")
                    elif abs(kurt) < 1:
                        st.warning("Curtose moderadamente diferente da normal")
                    else:
                        st.error("Curtose muito diferente da normal")
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab5:
        st.subheader("🔍 Gráficos de Dispersão")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        
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
                
                # Cálculo da correlação
                correlacao = dados[eixo_x].corr(dados[eixo_y])
                st.metric("Coeficiente de Correlação", f"{correlacao:.3f}")
                
                # Interpretação da correlação
                if abs(correlacao) > 0.7:
                    st.success("Correlação forte")
                elif abs(correlacao) > 0.3:
                    st.info("Correlação moderada")
                else:
                    st.warning("Correlação fraca")
        else:
            st.warning("❌ Número insuficiente de colunas numéricas")

    with tab6:
        st.subheader("📦 Boxplots por Variável")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para boxplot:", colunas_numericas, key="boxplot_col")
            
            if coluna_selecionada:
                fig = px.box(dados, y=coluna_selecionada, 
                            title=f"Boxplot de {coluna_selecionada}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Identificar outliers
                Q1 = dados[coluna_selecionada].quantile(0.25)
                Q3 = dados[coluna_selecionada].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = dados[(dados[coluna_selecionada] < lower_bound) | 
                                (dados[coluna_selecionada] > upper_bound)]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Número de Outliers", len(outliers))
                with col2:
                    st.metric("% Outliers", f"{(len(outliers)/len(dados))*100:.1f}%")
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab7:
        st.subheader("🧮 Estatísticas Avançadas")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para análise avançada:", colunas_numericas, key="advanced_col")
            
            if coluna_selecionada:
                data = dados[coluna_selecionada].dropna()
                
                # Estatísticas avançadas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Soma", f"{data.sum():.2f}")
                with col2:
                    st.metric("Variância", f"{data.var():.2f}")
                with col3:
                    st.metric("Erro Padrão", f"{data.std()/np.sqrt(len(data)):.2f}")
                with col4:
                    st.metric("Intervalo", f"{data.max() - data.min():.2f}")
                
                # Intervalo de confiança
                st.subheader("📊 Intervalo de Confiança 95%")
                lower, upper = calcular_intervalo_confianca(data)
                st.metric("Média ± IC 95%", f"{data.mean():.2f} ± {(upper-lower)/2:.2f}")
                st.write(f"**Intervalo:** [{lower:.2f}, {upper:.2f}]")
                
                # Teste t simplificado
                st.subheader("📋 Teste t Simplificado")
                t_stat = teste_t_simplificado(data)
                st.metric("Estatística t", f"{t_stat:.3f}")
                
                # Interpretação do teste t
                if abs(t_stat) > 1.96:
                    st.success("Resultado estatisticamente significativo (|t| > 1.96)")
                else:
                    st.warning("Resultado não estatisticamente significativo (|t| ≤ 1.96)")

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

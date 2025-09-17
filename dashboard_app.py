# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os
from scipy import stats

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
                coluna_data = st.selectbox("Coluna de Data:", colunas_data)
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas)
            
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
                    st.metric("Tendência", "↗️" if dados[coluna_valor].iloc[-1] > dados[coluna_valor].iloc[0] else "↘️")
        else:
            st.warning("❌ Dados insuficientes para análise temporal")

    with tab2:
        st.subheader("📋 Estatísticas Descritivas por Coluna")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Selecione a coluna:", colunas_numericas)
            
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
                    st.metric("Coef. Variação", f"{(stats['std']/stats['mean'])*100:.1f}%")
                
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
            coluna_selecionada = st.selectbox("Coluna para análise de distribuição:", colunas_numericas)
            
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
                
                # Teste de normalidade
                st.subheader("📋 Teste de Normalidade")
                from scipy.stats import shapiro
                
                data_clean = dados[coluna_selecionada].dropna()
                if len(data_clean) > 3:
                    stat, p_value = shapiro(data_clean)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Estatística Shapiro-Wilk", f"{stat:.3f}")
                    with col4:
                        st.metric("Valor-p", f"{p_value:.3f}")
                        
                    if p_value > 0.05:
                        st.success("✅ Distribuição normal (p > 0.05)")
                    else:
                        st.warning("❌ Distribuição não normal (p ≤ 0.05)")
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab5:
        st.subheader("🔍 Gráficos de Dispersão")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(colunas_numericas) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X:", colunas_numericas)
            with col2:
                eixo_y = st.selectbox("Eixo Y:", colunas_numericas)
            
            if eixo_x and eixo_y:
                fig = px.scatter(dados, x=eixo_x, y=eixo_y, 
                                title=f"{eixo_y} vs {eixo_x}",
                                trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cálculo da correlação
                correlacao = dados[eixo_x].corr(dados[eixo_y])
                st.metric("Coeficiente de Correlação", f"{correlacao:.3f}")
        else:
            st.warning("❌ Número insuficiente de colunas numéricas")

    with tab6:
        st.subheader("📦 Boxplots por Variável")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para boxplot:", colunas_numericas)
            
            if coluna_selecionada:
                fig = px.box(dados, y=coluna_selecionada, 
                            title=f"Boxplot de {coluna_selecionada}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Identificar outliers
                Q1 = dados[coluna_selecionada].quantile(0.25)
                Q3 = dados[coluna_selecionada].quantile(0.75)
                IQR = Q3 - Q1
                outliers = dados[(dados[coluna_selecionada] < (Q1 - 1.5 * IQR)) | 
                                (dados[coluna_selecionada] > (Q3 + 1.5 * IQR))]
                
                st.metric("Número de Outliers", len(outliers))
        else:
            st.warning("❌ Nenhuma coluna numérica encontrada")

    with tab7:
        st.subheader("🧮 Estatísticas Avançadas")
        
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            coluna_selecionada = st.selectbox("Coluna para análise avançada:", colunas_numericas)
            
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
                
                # Testes estatísticos
                st.subheader("📊 Testes Estatísticos")
                
                # Teste t para média ≠ 0
                t_stat, p_value = stats.ttest_1samp(data, 0)
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Estatística t", f"{t_stat:.3f}")
                with col6:
                    st.metric("Valor-p (t-test)", f"{p_value:.3f}")
                
                # Intervalo de confiança 95%
                confidence = 0.95
                n = len(data)
                m = data.mean()
                se = data.std() / np.sqrt(n)
                h = se * stats.t.ppf((1 + confidence) / 2., n-1)
                
                st.metric("IC 95%", f"[{m-h:.2f}, {m+h:.2f}]")

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

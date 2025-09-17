# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Função para converter para data
def converter_para_data(coluna):
    """Tenta converter uma coluna para datetime"""
    try:
        return pd.to_datetime(coluna, dayfirst=True, errors='coerce')
    except:
        return coluna

# Função para detectar outliers
def detectar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dados[(dados[coluna] < lower_bound) | (dados[coluna] > upper_bound)]

# Função para calcular regressão linear manualmente
def calcular_regressao_linear(x, y):
    """Calcula regressão linear manualmente"""
    # Remover valores NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return None, None, None
    
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    y_mean = np.mean(y_clean)
    
    numerator = np.sum((x_clean - x_mean) * (y_clean - y_mean))
    denominator = np.sum((x_clean - x_mean) ** 2)
    
    if denominator == 0:
        return None, None, None
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calcular R²
    y_pred = slope * x_clean + intercept
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return slope, intercept, r_squared

def main():
    st.title("📊 Dashboard de Utilidades - Análise Completa")
    
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

    # Processar dados
    dados_processados = dados.copy()
    colunas_numericas = dados_processados.select_dtypes(include=[np.number]).columns.tolist()
    
    # Detectar colunas de data
    colunas_data = []
    for col in dados_processados.columns:
        if any(palavra in col.lower() for palavra in ['data', 'date', 'dia', 'time']):
            colunas_data.append(col)
            dados_processados[col] = converter_para_data(dados_processados[col])

    # Sidebar para filtros globais
    with st.sidebar:
        st.header("🎛️ Filtros Globais")
        
        # Filtro de período
        if colunas_data:
            coluna_data_filtro = st.selectbox("Coluna de data para filtro:", colunas_data)
            if pd.api.types.is_datetime64_any_dtype(dados_processados[coluna_data_filtro]):
                min_date = dados_processados[coluna_data_filtro].min()
                max_date = dados_processados[coluna_data_filtro].max()
                
                date_range = st.date_input(
                    "Selecione o período:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    dados_processados = dados_processados[
                        (dados_processados[coluna_data_filtro] >= pd.Timestamp(start_date)) &
                        (dados_processados[coluna_data_filtro] <= pd.Timestamp(end_date))
                    ]
        
        # Filtro de outliers
        st.subheader("🔍 Gerenciamento de Outliers")
        remover_outliers = st.checkbox("Remover outliers automaticamente")
        
        if remover_outliers and colunas_numericas:
            coluna_outliers = st.selectbox("Coluna para análise de outliers:", colunas_numericas)
            if coluna_outliers:
                outliers = detectar_outliers(dados_processados, coluna_outliers)
                st.info(f"📊 {len(outliers)} outliers detectados")
                
                if st.button("Remover outliers"):
                    dados_processados = dados_processados[~dados_processados.index.isin(outliers.index)]
                    st.success("Outliers removidos!")

    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Análise de Séries Temporais", 
        "📊 Estatística Detalhada", 
        "🔥 Análise de Correlações", 
        "🔍 Gráficos de Dispersão"
    ])

    with tab1:
        st.header("📈 Análise de Séries Temporais")
        
        if colunas_data and colunas_numericas:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                coluna_data = st.selectbox("Coluna de Data:", colunas_data, key="temp_data")
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas, key="temp_valor")
            with col3:
                tipo_grafico = st.selectbox("Tipo de Gráfico:", 
                                           ["Linha", "Área", "Barra", "Scatter"])
            
            if coluna_data and coluna_valor:
                dados_temp = dados_processados.sort_values(by=coluna_data)
                
                # Criar gráfico baseado no tipo selecionado
                if tipo_grafico == "Linha":
                    fig = px.line(dados_temp, x=coluna_data, y=coluna_valor, 
                                 title=f"Evolução Temporal de {coluna_valor}")
                elif tipo_grafico == "Área":
                    fig = px.area(dados_temp, x=coluna_data, y=coluna_valor,
                                 title=f"Evolução Temporal de {coluna_valor}")
                elif tipo_grafico == "Barra":
                    fig = px.bar(dados_temp, x=coluna_data, y=coluna_valor,
                                title=f"Evolução Temporal de {coluna_valor}")
                else:  # Scatter
                    fig = px.scatter(dados_temp, x=coluna_data, y=coluna_valor,
                                    title=f"Relação Temporal de {coluna_valor}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas temporais
                st.subheader("📊 Estatísticas Temporais")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{dados_temp[coluna_valor].mean():.2f}")
                    st.metric("Mediana", f"{dados_temp[coluna_valor].median():.2f}")
                with col2:
                    st.metric("Desvio Padrão", f"{dados_temp[coluna_valor].std():.2f}")
                    st.metric("Variância", f"{dados_temp[coluna_valor].var():.2f}")
                with col3:
                    st.metric("Mínimo", f"{dados_temp[coluna_valor].min():.2f}")
                    st.metric("Máximo", f"{dados_temp[coluna_valor].max():.2f}")
                with col4:
                    Q1 = dados_temp[coluna_valor].quantile(0.25)
                    Q3 = dados_temp[coluna_valor].quantile(0.75)
                    st.metric("Q1 (25%)", f"{Q1:.2f}")
                    st.metric("Q3 (75%)", f"{Q3:.2f}")

    with tab2:
        st.header("📊 Estatística Detalhada")
        
        if colunas_numericas:
            coluna_analise = st.selectbox("Selecione a coluna para análise:", colunas_numericas, key="stats_col")
            
            if coluna_analise:
                # Estatísticas básicas
                st.subheader("Estatísticas Descritivas")
                stats_data = dados_processados[coluna_analise].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                metrics = [
                    ("Média", stats_data['mean']),
                    ("Mediana", stats_data['50%']),
                    ("Moda", dados_processados[coluna_analise].mode().iloc[0] if not dados_processados[coluna_analise].mode().empty else np.nan),
                    ("Desvio Padrão", stats_data['std']),
                    ("Variância", stats_data['std']**2),
                    ("Coef. Variação", (stats_data['std']/stats_data['mean'])*100 if stats_data['mean'] != 0 else 0),
                    ("Mínimo", stats_data['min']),
                    ("Máximo", stats_data['max']),
                    ("Amplitude", stats_data['max'] - stats_data['min']),
                    ("Q1 (25%)", stats_data['25%']),
                    ("Q3 (75%)", stats_data['75%']),
                    ("IQR", stats_data['75%'] - stats_data['25%'])
                ]
                
                for i, (name, value) in enumerate(metrics):
                    with [col1, col2, col3, col4][i % 4]:
                        if not np.isnan(value):
                            st.metric(name, f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                
                # Análise de distribuição
                st.subheader("📈 Análise de Distribuição")
                
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Coeficientes de forma
                    skewness = dados_processados[coluna_analise].skew()
                    kurtosis = dados_processados[coluna_analise].kurtosis()
                    
                    st.write("**Análise de Distribuição:**")
                    st.write(f"Assimetria: {skewness:.3f}")
                    st.write(f"Curtose: {kurtosis:.3f}")
                    
                    if abs(skewness) < 0.5:
                        st.success("Distribuição aproximadamente simétrica")
                    elif abs(skewness) < 1:
                        st.warning("Distribuição moderadamente assimétrica")
                    else:
                        st.error("Distribuição fortemente assimétrica")
                
                with dist_col2:
                    # Histograma
                    fig = px.histogram(dados_processados, x=coluna_analise, 
                                      title=f"Distribuição de {coluna_analise}",
                                      nbins=30)
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🔥 Análise de Correlações")
        
        if len(colunas_numericas) > 1:
            # Selecionar variáveis para correlação
            st.subheader("Seleção de Variáveis")
            variaveis_selecionadas = st.multiselect(
                "Selecione as variáveis para análise de correlação:",
                options=colunas_numericas,
                default=colunas_numericas[:min(5, len(colunas_numericas))]
            )
            
            if len(variaveis_selecionadas) > 1:
                # Matriz de correlação
                corr_matrix = dados_processados[variaveis_selecionadas].corr()
                
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**10 Maiores Correlações:**")
                    top_correlations = df_corr.nlargest(10, 'Abs_Correlation')
                    for _, row in top_correlations.iterrows():
                        corr_color = "🟢" if row['Correlação'] > 0 else "🔴"
                        st.write(f"{corr_color} {row['Variável 1']} ↔ {row['Variável 2']}: **{row['Correlação']:.3f}**")
                
                with col2:
                    st.write("**10 Menores Correlações:**")
                    bottom_correlations = df_corr.nsmallest(10, 'Abs_Correlation')
                    for _, row in bottom_correlations.iterrows():
                        corr_color = "🟢" if row['Correlação'] > 0 else "🔴"
                        st.write(f"{corr_color} {row['Variável 1']} ↔ {row['Variável 2']}: **{row['Correlação']:.3f}**")

    with tab4:
        st.header("🔍 Gráficos de Dispersão com Regressão")
        
        if len(colunas_numericas) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X:", colunas_numericas, key="scatter_x")
            with col2:
                eixo_y = st.selectbox("Eixo Y:", colunas_numericas, key="scatter_y")
            
            if eixo_x and eixo_y:
                # Gráfico de dispersão SEM trendline (que causa o erro)
                fig = px.scatter(dados_processados, x=eixo_x, y=eixo_y, 
                                title=f"{eixo_y} vs {eixo_x}")
                
                # Calcular regressão linear manualmente
                slope, intercept, r_squared = calcular_regressao_linear(
                    dados_processados[eixo_x].values,
                    dados_processados[eixo_y].values
                )
                
                # Adicionar linha de regressão manualmente se possível
                if slope is not None and intercept is not None:
                    x_range = np.linspace(dados_processados[eixo_x].min(), dados_processados[eixo_x].max(), 100)
                    y_pred = slope * x_range + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name='Linha de Regressão',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Adicionar equação da reta
                    equation = f"y = {slope:.2f}x + {intercept:.2f}"
                    r2_text = f"R² = {r_squared:.3f}"
                    
                    fig.add_annotation(
                        x=0.05, y=0.95,
                        xref="paper", yref="paper",
                        text=f"{equation}<br>{r2_text}",
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas de correlação
                correlacao = dados_processados[eixo_x].corr(dados_processados[eixo_y])
                st.metric("Coeficiente de Correlação de Pearson", f"{correlacao:.3f}")
                
                if r_squared is not None:
                    st.metric("Coeficiente de Determinação (R²)", f"{r_squared:.3f}")
                
                # Interpretação
                if abs(correlacao) > 0.7:
                    st.success("Correlação forte")
                elif abs(correlacao) > 0.3:
                    st.info("Correlação moderada")
                else:
                    st.warning("Correlação fraca")

    # Download dos dados processados
    st.sidebar.header("💾 Exportar Dados")
    csv = dados_processados.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Baixar dados processados",
        data=csv,
        file_name="dados_processados.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

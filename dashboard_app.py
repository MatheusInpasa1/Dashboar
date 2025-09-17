# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="KPI Utilidades Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        caminho = r'C:\Users\matheus.mendes\Desktop\Dashboard\KPI - Utilidades 101623 rev5.xlsx'
        df = pd.read_excel(caminho, header=0)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# Função para converter datas no formato brasileiro
def converter_data_brasileira(coluna_series):
    """Converte datas no formato dd/mm/yyyy para datetime"""
    try:
        # Primeiro tenta converter como datetime normal
        return pd.to_datetime(coluna_series, dayfirst=True, errors='coerce')
    except:
        return coluna_series

# Carregar dados
df = load_data()

if df.empty:
    st.stop()

# Identificar coluna de data automaticamente
colunas_data = []
for col in df.columns:
    col_str = str(col).lower()
    if any(palavra in col_str for palavra in ['data', 'date', 'dia', 'time', 'hora', 'timestamp', 'amostra']):
        colunas_data.append(col)
    elif df[col].dtype == 'object':
        # Testar se é data
        try:
            teste = pd.to_datetime(df[col].head(), dayfirst=True, errors='coerce')
            if not teste.isna().all():
                colunas_data.append(col)
        except:
            pass

# Sidebar - FILTROS
with st.sidebar:
    st.header("⚙️ Filtros Avançados")
    
    # 1. FILTROS POR COLUNAS CATEGÓRICAS
    st.subheader("🔍 Filtros por Categoria")
    colunas_categoricas = [col for col in df.columns if df[col].nunique() < 20 and df[col].nunique() > 1]
    
    filtros_aplicados = {}
    for coluna in colunas_categoricas[:5]:
        valores = ['Todos'] + sorted(df[coluna].dropna().unique().tolist())
        selecao = st.selectbox(f"Filtrar {coluna}:", valores)
        filtros_aplicados[coluna] = selecao
    
    # 2. FILTRO POR DATA - CORRIGIDO
    st.subheader("📅 Filtro Temporal")
    
    data_filter_container = st.container()
    
    with data_filter_container:
        if colunas_data:
            coluna_data_selecionada = st.selectbox(
                "Selecione a coluna de data:",
                options=colunas_data
            )
            
            # Converter para datetime com formato brasileiro
            df[coluna_data_selecionada] = converter_data_brasileira(df[coluna_data_selecionada])
            
            # Verificar se a conversão funcionou
            if pd.api.types.is_datetime64_any_dtype(df[coluna_data_selecionada]):
                datas_validas = df[coluna_data_selecionada].dropna()
                
                if not datas_validas.empty:
                    data_min = datas_validas.min()
                    data_max = datas_validas.max()
                    
                    st.success(f"✅ Datas de {data_min.strftime('%d/%m/%Y')} a {data_max.strftime('%d/%m/%Y')}")
                    
                    # Widgets de data com formato brasileiro
                    data_inicio = st.date_input(
                        "Data início:",
                        value=data_min,
                        min_value=data_min,
                        max_value=data_max
                    )
                    
                    data_fim = st.date_input(
                        "Data fim:",
                        value=data_max,
                        min_value=data_min,
                        max_value=data_max
                    )
                    
                    # Aplicar filtro de data
                    aplicar_filtro_data = st.checkbox("Aplicar filtro de data", value=True)
                    
                    if aplicar_filtro_data:
                        data_inicio_dt = pd.to_datetime(data_inicio)
                        data_fim_dt = pd.to_datetime(data_fim)
                    else:
                        data_inicio_dt = None
                        data_fim_dt = None
                else:
                    st.warning("Não há datas válidas para filtro")
                    data_inicio_dt = None
                    data_fim_dt = None
            else:
                st.warning("Coluna selecionada não contém datas válidas")
                data_inicio_dt = None
                data_fim_dt = None
        else:
            st.info("ℹ️ Nenhuma coluna de data identificada")
            coluna_data_selecionada = None
            data_inicio_dt = None
            data_fim_dt = None

# APLICAR FILTROS - CORRIGIDO
df_filtrado = df.copy()

# 1. Aplicar filtros categóricos
for coluna, valor in filtros_aplicados.items():
    if valor != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[coluna] == valor]

# 2. Aplicar filtro de data - CORREÇÃO IMPORTANTE
if (coluna_data_selecionada and 
    data_inicio_dt is not None and 
    data_fim_dt is not None and
    pd.api.types.is_datetime64_any_dtype(df_filtrado[coluna_data_selecionada])):
    
    df_filtrado = df_filtrado[
        (df_filtrado[coluna_data_selecionada] >= data_inicio_dt) & 
        (df_filtrado[coluna_data_selecionada] <= data_fim_dt)
    ]

# Restante do código continua igual...
# Interface principal
st.title("📊 Dashboard KPI Utilidades - Análise Avançada")
st.markdown(f"**📈 {len(df_filtrado)} registros** | **🏷️ {len(df.columns)} parâmetros**")

# KPIs Rápidos
if not df_filtrado.empty:
    st.header("📈 Métricas do Período")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Amostras", len(df_filtrado))
    with col2:
        st.metric("Taxa de Uso", f"{(len(df_filtrado)/len(df)*100):.1f}%")
    with col3:
        st.metric("Colunas", len(df.columns))
    with col4:
        missing = df_filtrado.isnull().sum().sum()
        st.metric("Dados Faltantes", missing)

# SELEÇÃO DE PARÂMETROS PARA GRÁFICOS
st.header("🎯 Seleção de Parâmetros")
colunas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    parametros_grafico1 = st.multiselect(
        "Parâmetros para Gráfico Temporal:",
        options=colunas_numericas,
        default=colunas_numericas[:2] if len(colunas_numericas) >= 2 else colunas_numericas[:1]
    )
with col2:
    parametros_correlacao = st.multiselect(
        "Parâmetros para Análise de Correlação:",
        options=colunas_numericas,
        default=colunas_numericas[:4] if len(colunas_numericas) >= 4 else colunas_numericas
    )

# GRÁFICOS EM ABAS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal", "📊 Correlação", "📋 Distribuição", "🧮 Estatísticas"])

with tab1:
    st.header("📈 Análise Temporal")
    if coluna_data_selecionada and parametros_grafico1 and not df_filtrado.empty:
        fig_temporal = px.line(df_filtrado, x=coluna_data_selecionada, y=parametros_grafico1,
                              title="Evolução Temporal dos Parâmetros")
        fig_temporal.update_layout(
            height=500,
            xaxis_title="Data",
            yaxis_title="Valores",
            xaxis=dict(tickformat='%d/%m/%Y')  # FORMATO BRASILEIRO
        )
        st.plotly_chart(fig_temporal, use_container_width=True)
    else:
        st.warning("Selecione parâmetros e verifique a coluna de data")

with tab2:
    st.header("📊 Análise de Correlação")
    if len(parametros_correlacao) >= 2 and not df_filtrado.empty:
        corr_matrix = df_filtrado[parametros_correlacao].corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corr = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto",
                                title="Matriz de Correlação",
                                color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            if len(parametros_correlacao) <= 4:
                fig_scatter = px.scatter_matrix(df_filtrado[parametros_correlacao],
                                              title="Matrix de Dispersão")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top correlações
        st.subheader("🔝 Principais Correlações")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'Parâmetro 1': corr_matrix.columns[i],
                    'Parâmetro 2': corr_matrix.columns[j],
                    'Correlação': corr_matrix.iloc[i, j]
                })
        
        df_corr = pd.DataFrame(correlations)
        df_corr['Abs_Correlação'] = df_corr['Correlação'].abs()
        df_corr = df_corr.sort_values('Abs_Correlação', ascending=False).head(10)
        st.dataframe(df_corr[['Parâmetro 1', 'Parâmetro 2', 'Correlação']].round(3))
        
    else:
        st.warning("Selecione pelo menos 2 parâmetros para análise de correlação")

# ... (restante do código das outras abas)

# Tabela de dados com datas formatadas
st.header("📋 Dados Filtrados")
if not df_filtrado.empty and coluna_data_selecionada:
    # Formatar datas para exibição
    df_display = df_filtrado.copy()
    if pd.api.types.is_datetime64_any_dtype(df_display[coluna_data_selecionada]):
        df_display[coluna_data_selecionada] = df_display[coluna_data_selecionada].dt.strftime('%d/%m/%Y')
    
    st.dataframe(df_display, use_container_width=True, height=300)
else:
    st.dataframe(df_filtrado, use_container_width=True, height=300)

# Download
with st.sidebar:
    st.header("💾 Exportar Dados")
    csv = df_filtrado.to_csv(index=False, date_format='%d/%m/%Y')  # Formato brasileiro
    st.download_button(
        "📥 Baixar CSV filtrado",
        data=csv,
        file_name="dados_filtrados.csv",
        mime="text/csv"
    )
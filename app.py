import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard 3D de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard 3D de Manutenção Preditiva com DBSCAN")
st.markdown("""
Visualização tridimensional interativa de sensores de chão de fábrica. 
Passe o mouse sobre os pontos para inspecionar os valores e utilize os filtros abaixo.
""")

# ==========================================
# 1. GERAÇÃO DOS DADOS (3 MÉTRICAS PARA O 3D)
# ==========================================
@st.cache_data
def gerar_dados(seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Funcionamento normal - Operação Padrão
    vib1 = np.random.normal(loc=50, scale=4, size=150)
    temp1 = np.random.normal(loc=65, scale=3, size=150)
    pressao1 = np.random.normal(loc=30, scale=2, size=150)
    
    # Cluster 2: Funcionamento normal - Carga Elevada
    vib2 = np.random.normal(loc=80, scale=5, size=100)
    temp2 = np.random.normal(loc=85, scale=4, size=100)
    pressao2 = np.random.normal(loc=50, scale=3, size=100)
    
    # Outliers / Anomalias intencionais (espalhados)
    vib_out = np.random.uniform(low=15, high=110, size=25)
    temp_out = np.random.uniform(low=35, high=105, size=25)
    pressao_out = np.random.uniform(low=10, high=70, size=25)
    
    # Unificando
    vibracao = np.concatenate([vib1, vib2, vib_out])
    temperatura = np.concatenate([temp1, temp2, temp_out])
    pressao = np.concatenate([pressao1, pressao2, pressao_out])
    
    id_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'ID_Maquina': id_maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura,
        'Pressao_Oleo': pressao
    })
    
    return df

df = gerar_dados()

# ==========================================
# 2. BARRA LATERAL (CONTROLES DO DBSCAN)
# ==========================================
st.sidebar.header("⚙️ Parâmetros do DBSCAN")

eps = st.sidebar.slider(
    "Distância Máxima (eps)",
    min_value=2.0,
    max_value=20.0,
    value=7.5,
    step=0.5,
    help="Define a distância máxima 3D entre pontos vizinhos."
)

min_samples = st.sidebar.slider(
    "Amostras Mínimas (min_samples)",
    min_value=2,
    max_value=15,
    value=4,
    step=1,
    help="Número mínimo de pontos para formar um cluster."
)

# ==========================================
# 3. APLICAÇÃO DO DBSCAN
# ==========================================
X = df[['Frequencia_Vibracao', 'Pico_Temperatura', 'Pressao_Oleo']]

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['Cluster'] = dbscan.fit_predict(X)

# Criando coluna descritiva de status para facilitar filtros
df['Status'] = df['Cluster'].apply(lambda c: 'Anomalia (Outlier)' if c == -1 else f'Cluster Normal {c}')

num_outliers = (df['Cluster'] == -1).sum()
num_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)

# Métricas rápidas no painel
col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Normais", num_clusters)
col3.metric("Anomalias Detectadas", num_outliers, delta_color="inverse")

# ==========================================
# 4. GRÁFICO 3D INTERATIVO (PLOTLY)
# ==========================================
st.subheader("🌐 Modelo 3D: Vibração x Temperatura x Pressão")

# Definir cores de alto contraste (Outlier em preto estrito, demais com cores vivas)
cores_mapeadas = {}
unique_clusters = sorted(df['Cluster'].unique())
vibrant_colors = ['#00FFCC', '#FF007F', '#FFD700', '#007FFF', '#FF4500']

for idx, c in enumerate(unique_clusters):
    if c == -1:
        cores_mapeadas['Anomalia (Outlier)'] = '#000000' # Preto de alto contraste
    else:
        cores_mapeadas[f'Cluster Normal {c}'] = vibrant_colors[idx % len(vibrant_colors)]

fig = px.scatter_3d(
    df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    z='Pressao_Oleo',
    color='Status',
    color_discrete_map=cores_mapeadas,
    hover_name='ID_Maquina',
    hover_data={
        'Frequencia_Vibracao': ':.2f Hz',
        'Pico_Temperatura': ':.2f °C',
        'Pressao_Oleo': ':.2f PSI',
        'Cluster': False,
        'Status': True
    },
    opacity=0.85
)

# Removendo legenda e ajustando o layout 3D
fig.update_layout(
    showlegend=False,  # Sem legenda conforme solicitado
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='Frequência de Vibração (Hz)',
        yaxis_title='Pico de Temperatura (°C)',
        zaxis_title='Pressão do Óleo (PSI)'
    ),
    height=650
)

# Aumentar tamanho dos marcadores 3D
fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. TABELA INTERATIVA COM FILTROS (NORMAIS E ANOMALIAS)
# ==========================================
st.markdown("---")
st.subheader("📋 Tabela de Dados e Filtragem de Máquinas")

# Filtro interativo para o usuário escolher o que deseja visualizar na tabela
filtro_status = st.selectbox(
    "Filtrar por Status do Equipamento:",
    options=["Todas as Máquinas", "Apenas Anomalias (Outliers)", "Apenas Máquinas Normais"]
)

# Aplicar o filtro na tabela
df_exibicao = df.copy()
if filtro_status == "Apenas Anomalias (Outliers)":
    df_exibicao = df_exibicao[df_exibicao['Cluster'] == -1]
elif filtro_status == "Apenas Máquinas Normais":
    df_exibicao = df_exibicao[df_exibicao['Cluster'] != -1]

# Exibir tabela formatada
st.dataframe(
    df_exibicao[['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura', 'Pressao_Oleo', 'Status']].style.format({
        'Frequencia_Vibracao': '{:.2f} Hz',
        'Pico_Temperatura': '{:.2f} °C',
        'Pressao_Oleo': '{:.2f} PSI'
    }),
    use_container_width=True
)

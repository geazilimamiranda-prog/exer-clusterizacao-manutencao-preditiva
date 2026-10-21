import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import DBSCAN

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva com DBSCAN")
st.markdown("""
Esta aplicação simula a detecção de anomalias em sensores de chão de fábrica usando o algoritmo **DBSCAN**.
Ajuste os parâmetros na barra lateral para recalibrar a detecção de falhas/outliers.
""")

# ==========================================
# 1. GERAÇÃO DOS DADOS (COM OUTLIERS)
# ==========================================
@st.cache_data
def gerar_dados(seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Funcionamento normal - Operação Padrão
    vib_normal1 = np.random.normal(loc=50, scale=5, size=150)
    temp_normal1 = np.random.normal(loc=65, scale=4, size=150)
    
    # Cluster 2: Funcionamento normal - Carga Elevada
    vib_normal2 = np.random.normal(loc=80, scale=6, size=100)
    temp_normal2 = np.random.normal(loc=85, scale=5, size=100)
    
    # Outliers / Anomalias intencionais
    vib_outliers = np.random.uniform(low=20, high=120, size=20)
    temp_outliers = np.random.uniform(low=40, high=110, size=20)
    
    # Unificando os dados
    vibracao = np.concatenate([vib_normal1, vib_normal2, vib_outliers])
    temperatura = np.concatenate([temp_normal1, temp_normal2, temp_outliers])
    
    # Criando identificadores de máquinas
    id_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'ID_Maquina': id_maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura
    })
    
    return df

df = gerar_dados()

# ==========================================
# 2. BARRA LATERAL (CONTROLES DO DBSCAN)
# ==========================================
st.sidebar.header("⚙️ Parâmetros do DBSCAN")

eps = st.sidebar.slider(
    "Distância Máxima (eps)",
    min_value=1.0,
    max_value=15.0,
    value=5.5,
    step=0.5,
    help="Define a distância máxima entre dois pontos para que sejam considerados vizinhos."
)

min_samples = st.sidebar.slider(
    "Amostras Mínimas (min_samples)",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar um cluster/grupo."
)

# ==========================================
# 3. APLICAÇÃO DO DBSCAN
# ==========================================
X = df[['Frequencia_Vibracao', 'Pico_Temperatura']]

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['Cluster'] = dbscan.fit_predict(X)

# No DBSCAN, o rótulo -1 representa os outliers (anomalias)
num_outliers = (df['Cluster'] == -1).sum()
num_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)

# Métricas rápidas no painel
col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Normais Identificados", num_clusters)
col3.metric("Anomalias/Outliers Detectados", num_outliers, delta_color="inverse")

# ==========================================
# 4. GRÁFICO DE DISPERSÃO COM SEABORN
# ==========================================
st.subheader("📊 Agrupamento de Sensores e Detecção de Anomalias")

# Definir paleta de cores: garantir que o cluster -1 (outliers) seja PRETO
clusters_unicos = sorted(df['Cluster'].unique())
palette = {}

# Paleta suave para os clusters normais
cores_suaves = sns.color_palette("tab10", len(clusters_unicos))

for idx, c in enumerate(clusters_unicos):
    if c == -1:
        palette[c] = "#000000"  # Preto para Outliers
    else:
        palette[c] = cores_suaves[idx]

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    hue='Cluster',
    style=(df['Cluster'] == -1),  # Estilo diferente (X) para outliers
    markers={False: 'o', True: 'X'},
    palette=palette,
    s=100,
    ax=ax
)

# Estilização do gráfico
ax.set_title("Frequência de Vibração vs. Pico de Temperatura", fontsize=14, pad=15)
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=11)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

# Legenda personalizada
handles, labels = ax.get_legend_handles_labels()
novas_legendas = []
for l in labels:
    if l == '-1':
        novas_legendas.append("Anomalia / Outlier")
    else:
        novas_legendas.append(f"Cluster Normal {l}")

ax.legend(handles=handles, labels=novas_legendas, title="Status do Equipamento", bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

# ==========================================
# 5. TABELA DE MÁQUINAS ANÔMALAS
# ==========================================
st.subheader("🚨 Máquinas Consideradas Anômalas")

df_anomalias = df[df['Cluster'] == -1][['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura']].reset_index(drop=True)

if not df_anomalias.empty:
    st.warning(f"Foram identificadas **{len(df_anomalias)}** máquinas operando fora dos padrões normais:")
    st.dataframe(
        df_anomalias.style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C'
        }),
        use_container_width=True
    )
else:
    st.success("Nenhuma anomalia foi detectada com os parâmetros atuais do DBSCAN.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------

st.set_page_config(
    page_title="Video Games Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CARGA DE DATOS (OPTIMIZADA)
# --------------------------------------------------

@st.cache_data(show_spinner=True)
def load_data():
    url = "https://github.com/yerquin15/Videojuegos-dashboard/releases/download/v1.0/normalized_dataset.csv"
    return pd.read_csv(url, low_memory=False)

df = load_data()

# --------------------------------------------------
# SIDEBAR - FILTROS
# --------------------------------------------------

st.sidebar.title("Filtros")

year = st.sidebar.selectbox(
    "Año de lanzamiento",
    sorted(df["release_year"].dropna().unique(), reverse=True)
)

age = st.sidebar.multiselect(
    "Clasificación ESRB",
    sorted(df["required_age"].dropna().unique()),
    default=sorted(df["required_age"].dropna().unique())
)

price_range = st.sidebar.slider(
    "Rango de precio",
    float(df["price"].min()),
    float(df["price"].max()),
    (0.0, float(df["price"].max()))
)

filtered = df[
    (df["release_year"] == year) &
    (df["required_age"].isin(age)) &
    (df["price"].between(price_range[0], price_range[1]))
].copy()

# --------------------------------------------------
# MÉTRICAS PRINCIPALES
# --------------------------------------------------

st.title("Dashboard de Videojuegos")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Número de juegos", len(filtered))
col2.metric("Precio promedio", f"${filtered['price'].mean():.2f}")
col3.metric("Porcentaje positivo promedio", f"{filtered['porcentaje_positive_total'].mean() * 100:.1f}%")
col4.metric("Tiempo promedio jugado", f"{filtered['average_playtime_forever'].mean():.1f} horas")

st.divider()

# --------------------------------------------------
# DASHBOARD 1: PRECIO VS VALORACIÓN
# --------------------------------------------------

st.subheader("Relación entre precio y valoración")

fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.scatter(
    filtered["price"],
    filtered["porcentaje_positive_total"],
    alpha=0.5
)
ax1.set_xlabel("Precio")
ax1.set_ylabel("Porcentaje positivo")
st.pyplot(fig1)

# --------------------------------------------------
# DASHBOARD 2: DISTRIBUCIÓN POR CLASIFICACIÓN ESRB
# --------------------------------------------------

st.subheader("Distribución por clasificación ESRB")

fig2, ax2 = plt.subplots(figsize=(5, 4))
filtered["required_age"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax2
)
ax2.set_ylabel("")
st.pyplot(fig2)

st.divider()

# --------------------------------------------------
# DASHBOARD 3: POPULARIDAD VS CALIDAD
# --------------------------------------------------

st.subheader("Popularidad vs calidad")

fig3, ax3 = plt.subplots(figsize=(5, 4))
ax3.scatter(
    filtered["total_num_reviews"],
    filtered["porcentaje_positive_total"],
    alpha=0.5
)
ax3.set_xlabel("Número de reseñas")
ax3.set_ylabel("Porcentaje positivo")
ax3.set_xscale("log")
st.pyplot(fig3)

st.divider()

# --------------------------------------------------
# DASHBOARD 4: EVOLUCIÓN TEMPORAL
# --------------------------------------------------

st.subheader("Evolución anual de la industria")

annual = (
    filtered.groupby("release_year")
    .agg(
        juegos=("price", "count"),
        precio_promedio=("price", "mean"),
        valoracion_promedio=("porcentaje_positive_total", "mean")
    )
    .reset_index()
)

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(annual["release_year"], annual["precio_promedio"], label="Precio promedio")
ax4.plot(annual["release_year"], annual["valoracion_promedio"], label="Valoración promedio")
ax4.set_xlabel("Año")
ax4.legend()
st.pyplot(fig4)

st.divider()

# --------------------------------------------------
# DASHBOARD 5: JUEGOS DESTACADOS POCO CONOCIDOS
# --------------------------------------------------

st.subheader("Juegos bien valorados con poca popularidad")

hidden_gems = filtered[
    (filtered["porcentaje_positive_total"] > 0.9) &
    (filtered["total_num_reviews"] < filtered["total_num_reviews"].quantile(0.25))
].sort_values("porcentaje_positive_total", ascending=False)

st.dataframe(
    hidden_gems[
        ["price", "total_num_reviews", "porcentaje_positive_total", "average_playtime_forever"]
    ].head(10),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# DASHBOARD 6: EXPLORADOR DINÁMICO
# --------------------------------------------------

st.subheader("Explorador dinámico de variables")

numeric_cols = filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()

selected_vars = st.multiselect(
    "Selecciona hasta 3 variables numéricas",
    numeric_cols,
    max_selections=3
)

if len(selected_vars) == 1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(filtered[selected_vars[0]].dropna(), bins=30)
    ax.set_xlabel(selected_vars[0])
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

elif len(selected_vars) == 2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        filtered[selected_vars[0]],
        filtered[selected_vars[1]],
        alpha=0.5
    )
    ax.set_xlabel(selected_vars[0])
    ax.set_ylabel(selected_vars[1])
    st.pyplot(fig)

elif len(selected_vars) == 3:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        filtered[selected_vars[0]],
        filtered[selected_vars[1]],
        filtered[selected_vars[2]],
        alpha=0.5
    )
    ax.set_xlabel(selected_vars[0])
    ax.set_ylabel(selected_vars[1])
    ax.set_zlabel(selected_vars[2])
    st.pyplot(fig)

else:
    st.info("Selecciona entre 1 y 3 variables para generar una visualización.")
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Videojuegos",
    layout="wide"
)

# --------------------------------------------------
# TEMA OSCURO (CSS)
# --------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #fafafa;
}
h1, h2, h3, h4 {
    color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    url = "https://github.com/yerquin15/Videojuegos-dashboard/releases/download/v1.0/normalized_dataset.csv"
    return pd.read_csv(url, low_memory=False)

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Filtros")

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
# TABS PRINCIPALES
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Visión general",
    "Análisis exploratorio",
    "Hallazgos y NLP"
])

# ==================================================
# TAB 1 - VISIÓN GENERAL
# ==================================================
with tab1:
    st.title("Dashboard de Videojuegos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Número de juegos", len(filtered))
    col2.metric("Precio promedio", f"${filtered['price'].mean():.2f}")
    col3.metric("Valoración promedio", f"{filtered['porcentaje_positive_total'].mean() * 100:.1f}%")
    col4.metric("Tiempo promedio jugado", f"{filtered['average_playtime_forever'].mean():.1f} hrs")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.scatter(
            filtered,
            x="price",
            y="porcentaje_positive_total",
            opacity=0.6,
            labels={
                "price": "Precio ($)",
                "porcentaje_positive_total": "Valoración positiva"
            },
            title="Precio vs Valoración"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.scatter(
            filtered,
            x="total_num_reviews",
            y="porcentaje_positive_total",
            opacity=0.6,
            log_x=True,
            labels={
                "total_num_reviews": "Número de reseñas (log)",
                "porcentaje_positive_total": "Valoración positiva"
            },
            title="Popularidad vs Calidad"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2 - ANÁLISIS EXPLORATORIO
# ==================================================
with tab2:
    st.header("Explorador dinámico de variables")

    numeric_cols = filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()

    selected_vars = st.multiselect(
        "Selecciona hasta 3 variables numéricas",
        numeric_cols,
        max_selections=3
    )

    if len(selected_vars) == 1:
        fig = px.histogram(
            filtered,
            x=selected_vars[0],
            nbins=30,
            title=f"Distribución de {selected_vars[0]}"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif len(selected_vars) == 2:
        fig = px.scatter(
            filtered,
            x=selected_vars[0],
            y=selected_vars[1],
            opacity=0.6,
            title=f"{selected_vars[0]} vs {selected_vars[1]}"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif len(selected_vars) == 3:
        fig = px.scatter_3d(
            filtered,
            x=selected_vars[0],
            y=selected_vars[1],
            z=selected_vars[2],
            opacity=0.6,
            title="Visualización 3D"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Selecciona entre 1 y 3 variables para visualizar.")

# ==================================================
# TAB 3 - HALLAZGOS Y NLP
# ==================================================
with tab3:
    st.header("Hallazgos relevantes")

    hidden_gems = filtered[
        (filtered["porcentaje_positive_total"] > 0.9) &
        (filtered["total_num_reviews"] < filtered["total_num_reviews"].quantile(0.25))
    ].sort_values("porcentaje_positive_total", ascending=False)

    st.subheader("Juegos muy bien valorados con baja popularidad")
    st.dataframe(
        hidden_gems[[
            "price",
            "total_num_reviews",
            "porcentaje_positive_total",
            "average_playtime_forever"
        ]].head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("WordCloud de descripciones / reseñas")

    text_column = st.selectbox(
        "Selecciona columna de texto",
        df.select_dtypes(include="object").columns
    )

    mask_image = st.file_uploader(
        "Opcional: sube una imagen para el WordCloud",
        type=["png", "jpg", "jpeg"]
    )

    text_data = " ".join(df[text_column].dropna().astype(str).values)

    if st.button("Generar WordCloud"):
        if mask_image:
            mask = np.array(Image.open(mask_image))
            wc = WordCloud(
                background_color="black",
                mask=mask,
                colormap="inferno",
                max_words=200
            ).generate(text_data)
        else:
            wc = WordCloud(
                background_color="black",
                colormap="inferno",
                max_words=200
            ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)




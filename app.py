# ==================================================
# IMPORTS
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud
from PIL import Image

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
st.set_page_config(
    page_title="Video Games Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_theme(style="dark")

# ==================================================
# CARGA DE DATOS
# ==================================================
@st.cache_data(show_spinner=True)
def load_data():
    url = "https://github.com/yerquin15/Videojuegos-dashboard/releases/download/v1.0/normalized_dataset.csv"
    return pd.read_csv(url, low_memory=False)

df = load_data()

# ==================================================
# SIDEBAR - FILTROS
# ==================================================
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

# ==================================================
# SECCIONES
# ==================================================
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
    col3.metric("Valoración promedio", f"{filtered['porcentaje_positive_total'].mean()*100:.1f}%")
    col4.metric("Tiempo promedio jugado", f"{filtered['average_playtime_forever'].mean():.1f} hrs")

    st.divider()

    fig_price_rating = px.scatter(
        filtered,
        x="price",
        y="porcentaje_positive_total",
        size="total_num_reviews",
        color="required_age",
        opacity=0.6,
        title="Precio vs valoración",
        template="plotly_dark"
    )
    st.plotly_chart(fig_price_rating, use_container_width=True)

    fig_popularity = px.scatter(
        filtered,
        x="total_num_reviews",
        y="porcentaje_positive_total",
        opacity=0.6,
        log_x=True,
        title="Popularidad vs calidad",
        template="plotly_dark"
    )
    st.plotly_chart(fig_popularity, use_container_width=True)

    annual = (
        df[
            df["required_age"].isin(age) &
            df["price"].between(price_range[0], price_range[1])
        ]
        .groupby("release_year")
        .agg(
            precio_promedio=("price", "mean"),
            valoracion_promedio=("porcentaje_positive_total", "mean")
        )
        .reset_index()
        .sort_values("release_year")
    )

    fig_trend = px.line(
        annual,
        x="release_year",
        y=["precio_promedio", "valoracion_promedio"],
        markers=True,
        title="Evolución anual de la industria",
        template="plotly_dark"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ==================================================
# TAB 2 - ANÁLISIS EXPLORATORIO
# ==================================================
with tab2:
    st.subheader("Distribución por clasificación ESRB")

    fig_esrb, ax_esrb = plt.subplots(figsize=(6, 4))
    filtered["required_age"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax_esrb
    )
    ax_esrb.set_ylabel("")
    st.pyplot(fig_esrb)

    st.divider()

    st.subheader("Explorador dinámico de variables")

    numeric_cols = filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()

    selected_vars = st.multiselect(
        "Selecciona hasta 3 variables numéricas",
        numeric_cols,
        max_selections=3
    )

    if len(selected_vars) == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(filtered[selected_vars[0]].dropna(), bins=30)
        ax.set_xlabel(selected_vars[0])
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    elif len(selected_vars) == 2:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                filtered[selected_vars[0]],
                filtered[selected_vars[1]],
                alpha=0.5
            )
            ax.set_xlabel(selected_vars[0])
            ax.set_ylabel(selected_vars[1])
            st.pyplot(fig)

        with col2:
            corr = filtered[selected_vars].corr()
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif len(selected_vars) == 3:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = plt.figure(figsize=(8, 6))
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

        with col2:
            corr = filtered[selected_vars].corr()
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# ==================================================
# TAB 3 - HALLAZGOS Y NLP
# ==================================================
with tab3:
    st.subheader("WordCloud")

    text_col = st.selectbox(
        "Selecciona columna de texto",
        [col for col in df.columns if df[col].dtype == "object"]
    )

    mask_file = st.file_uploader(
        "Opcional: subir imagen para forma del WordCloud",
        type=["png", "jpg", "jpeg"]
    )

    mask = None
    if mask_file:
        mask = np.array(Image.open(mask_file))

    text_data = " ".join(filtered[text_col].dropna().astype(str))

    if text_data.strip():
        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="viridis",
            mask=mask
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.divider()

    st.subheader("Hallazgos clave")

    uploaded_image = st.file_uploader(
        "Sube una imagen para acompañar el hallazgo",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, use_container_width=True)

    st.markdown("""
    - Juegos con alta valoración no siempre tienen alta popularidad  
    - El precio no es un predictor fuerte de calidad  
    - Existen juegos de nicho con excelente recepción  
    - El tiempo de juego promedio crece en títulos bien valorados  
    """)

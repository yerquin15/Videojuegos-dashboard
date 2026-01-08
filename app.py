import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Video Games Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
    }
    h1, h2, h3 {
        color: #00D4FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_theme(style="darkgrid", palette="muted")

# ==================================================
# CARGA DE DATOS
# ==================================================
@st.cache_data(show_spinner=True)
def load_data():
    try:
        url = "https://github.com/yerquin15/Videojuegos-dashboard/releases/download/v1.0/normalized_dataset.csv"
        df = pd.read_csv(url, low_memory=False)
        
        # Limpieza básica
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['porcentaje_positive_total'] = pd.to_numeric(df['porcentaje_positive_total'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ==================================================
# SIDEBAR - FILTROS
# ==================================================
st.sidebar.image("https://via.placeholder.com/300x100/0e1117/00D4FF?text=Gaming+Analytics", use_container_width=True)
st.sidebar.title(" Filtros de Análisis")
st.sidebar.markdown("---")

# Filtro de año
year_options = sorted([y for y in df["release_year"].dropna().unique() if not np.isnan(y)], reverse=True)
year = st.sidebar.selectbox(
    " Año de lanzamiento",
    year_options,
    index=0
)

# Filtro de clasificación ESRB
age_options = sorted(df["required_age"].dropna().unique())
age = st.sidebar.multiselect(
    " Clasificación ESRB",
    age_options,
    default=age_options
)

# Filtro de valoración mínima
min_rating = st.sidebar.slider(
    " Valoración mínima (%)",
    0.0,
    100.0,
    0.0,
    step=5.0
)


# Aplicar filtros
filtered = df[
    (df["release_year"] == year) &
    (df["required_age"].isin(age)) &
    (df["porcentaje_positive_total"] * 100 >= min_rating)
].copy()

# ==================================================
# TABS PRINCIPALES
# ==================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Visión General",
    " Análisis Exploratorio",
    " Tendencias Temporales",
    " Insights Avanzados",
    " Galería Visual"
])

# ==================================================
# TAB 1 - VISIÓN GENERAL
# ==================================================
with tab1:
    st.title(" Dashboard de Videojuegos")
    st.markdown(f"### Análisis del año **{int(year)}**")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            " Número de juegos",
            f"{len(filtered):,}",
            delta=f"{len(filtered) - len(df[df['release_year'] == year])}" if year else None
        )
    
    with col2:
        avg_price = filtered['price'].mean()
        st.metric(
            " Precio promedio",
            f"${avg_price:.2f}",
            delta=f"${avg_price - df['price'].mean():.2f}"
        )
    
    with col3:
        avg_rating = filtered['porcentaje_positive_total'].mean() * 100
        st.metric(
            " Valoración promedio",
            f"{avg_rating:.1f}%",
            delta=f"{avg_rating - (df['porcentaje_positive_total'].mean() * 100):.1f}%"
        )
    
    with col4:
        avg_playtime = filtered['average_playtime_forever'].mean()
        st.metric(
            " Tiempo promedio",
            f"{avg_playtime:.1f} hrs",
            delta=f"{avg_playtime - df['average_playtime_forever'].mean():.1f}"
        )
    
    st.markdown("---")
    
    # Gráficos principales
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader(" Relación Precio vs Valoración")
        fig_price_rating = px.scatter(
            filtered,
            x="price",
            y="porcentaje_positive_total",
            size="total_num_reviews",
            color="required_age",
            opacity=0.7,
            hover_data=['name'] if 'name' in filtered.columns else None,
            title="",
            template="plotly_dark",
            labels={
                'price': 'Precio ($)',
                'porcentaje_positive_total': 'Valoración (0-1)',
                'total_num_reviews': 'Número de Reviews',
                'required_age': 'Clasificación'
            }
        )
        fig_price_rating.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig_price_rating, use_container_width=True)
    
    with col_right:
        st.subheader(" Popularidad vs Calidad")
        fig_popularity = px.scatter(
            filtered,
            x="total_num_reviews",
            y="porcentaje_positive_total",
            opacity=0.7,
            log_x=True,
            hover_data=['name'] if 'name' in filtered.columns else None,
            title="",
            template="plotly_dark",
            color="price",
            color_continuous_scale="Viridis",
            labels={
                'total_num_reviews': 'Número de Reviews (log)',
                'porcentaje_positive_total': 'Valoración (0-1)',
                'price': 'Precio ($)'
            }
        )
        fig_popularity.update_layout(height=400)
        st.plotly_chart(fig_popularity, use_container_width=True)
    
    st.markdown("---")
    
    # Distribución de precios y valoraciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Distribución de Precios")
        fig_price_dist = go.Figure()
        fig_price_dist.add_trace(go.Histogram(
            x=filtered['price'],
            nbinsx=30,
            marker_color='#00D4FF',
            opacity=0.7,
            name='Frecuencia'
        ))
        fig_price_dist.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Precio ($)",
            yaxis_title="Frecuencia",
            showlegend=False
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        st.subheader(" Distribución de Valoraciones")
        fig_rating_dist = go.Figure()
        fig_rating_dist.add_trace(go.Histogram(
            x=filtered['porcentaje_positive_total'] * 100,
            nbinsx=20,
            marker_color='#FF6B6B',
            opacity=0.7,
            name='Frecuencia'
        ))
        fig_rating_dist.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Valoración (%)",
            yaxis_title="Frecuencia",
            showlegend=False
        )
        st.plotly_chart(fig_rating_dist, use_container_width=True)

# ==================================================
# TAB 2 - ANÁLISIS EXPLORATORIO
# ==================================================
with tab2:
    st.header(" Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Distribución ESRB")
        esrb_counts = filtered["required_age"].value_counts()
        
        fig_esrb = go.Figure(data=[go.Pie(
            labels=esrb_counts.index,
            values=esrb_counts.values,
            hole=0.4,
            marker_colors=px.colors.qualitative.Set3
        )])
        fig_esrb.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_esrb, use_container_width=True)
        
        # Estadísticas por ESRB
        st.markdown("** Stats por clasificación:**")
        for age_val in esrb_counts.index[:3]:
            subset = filtered[filtered['required_age'] == age_val]
            st.markdown(f"**{age_val}:** {len(subset)} juegos - Precio avg: ${subset['price'].mean():.2f}")
    
    with col2:
        st.subheader(" Top 10 Juegos Mejor Valorados")
        if 'name' in filtered.columns:
            top_games = filtered.nlargest(10, 'porcentaje_positive_total')[
                ['name', 'porcentaje_positive_total', 'price', 'total_num_reviews']
            ].copy()
            top_games['porcentaje_positive_total'] = top_games['porcentaje_positive_total'] * 100
            
            fig_top = px.bar(
                top_games,
                x='porcentaje_positive_total',
                y='name',
                orientation='h',
                color='price',
                color_continuous_scale='Turbo',
                template="plotly_dark",
                labels={
                    'porcentaje_positive_total': 'Valoración (%)',
                    'name': 'Juego',
                    'price': 'Precio ($)'
                }
            )
            fig_top.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
    
    st.markdown("---")
    
    # Explorador dinámico mejorado
    st.subheader(" Explorador Dinámico de Variables")
    
    numeric_cols = filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        selected_vars = st.multiselect(
            "Selecciona 2 o 3 variables numéricas para análisis",
            numeric_cols,
            default=[numeric_cols[0], numeric_cols[1]] if len(numeric_cols) >= 2 else None,
            max_selections=3
        )
    
    with col_right:
        color_var = st.selectbox(
            "Variable para color (opcional)",
            ["Ninguna"] + ['required_age'] if 'required_age' in filtered.columns else ["Ninguna"]
        )
    
    if len(selected_vars) == 2:
        color_param = None if color_var == "Ninguna" else color_var
        
        fig_scatter = px.scatter(
            filtered,
            x=selected_vars[0],
            y=selected_vars[1],
            color=color_param,
            opacity=0.6,
            template="plotly_dark",
            marginal_x="histogram",
            marginal_y="histogram"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlación
        if len(selected_vars) == 2:
            corr_val = filtered[selected_vars].corr().iloc[0, 1]
            st.info(f" **Correlación:** {corr_val:.3f}")
    
    elif len(selected_vars) == 3:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_3d = px.scatter_3d(
                filtered,
                x=selected_vars[0],
                y=selected_vars[1],
                z=selected_vars[2],
                color=color_var if color_var != "Ninguna" else None,
                opacity=0.7,
                template="plotly_dark"
            )
            fig_3d.update_layout(height=500)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # Matriz de correlación
            corr_matrix = filtered[selected_vars].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                height=300,
                title="Matriz de Correlación"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# ==================================================
# TAB 3 - TENDENCIAS TEMPORALES
# ==================================================
with tab3:
    st.header(" Análisis de Tendencias Temporales")
    
    # Preparar datos anuales
    annual = (
        df[
            (df["required_age"].isin(age)) &
        ]
        .groupby("release_year")
        .agg(
            num_juegos=("price", "count"),
            precio_promedio=("price", "mean"),
            valoracion_promedio=("porcentaje_positive_total", lambda x: x.mean() * 100),
            reviews_totales=("total_num_reviews", "sum"),
            tiempo_promedio=("average_playtime_forever", "mean")
        )
        .reset_index()
        .sort_values("release_year")
    )
    
    # Gráfico de evolución múltiple
    st.subheader(" Evolución de Métricas Clave")
    
    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Número de Juegos Lanzados", "Precio Promedio ($)", 
                       "Valoración Promedio (%)", "Tiempo de Juego Promedio (hrs)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Gráfico 1: Número de juegos
    fig_multi.add_trace(
        go.Scatter(x=annual["release_year"], y=annual["num_juegos"], 
                  mode='lines+markers', name='Juegos', line=dict(color='#00D4FF')),
        row=1, col=1
    )
    
    # Gráfico 2: Precio promedio
    fig_multi.add_trace(
        go.Scatter(x=annual["release_year"], y=annual["precio_promedio"], 
                  mode='lines+markers', name='Precio', line=dict(color='#4ECDC4')),
        row=1, col=2
    )
    
    # Gráfico 3: Valoración promedio
    fig_multi.add_trace(
        go.Scatter(x=annual["release_year"], y=annual["valoracion_promedio"], 
                  mode='lines+markers', name='Valoración', line=dict(color='#FF6B6B')),
        row=2, col=1
    )
    
    # Gráfico 4: Tiempo promedio
    fig_multi.add_trace(
        go.Scatter(x=annual["release_year"], y=annual["tiempo_promedio"], 
                  mode='lines+markers', name='Tiempo', line=dict(color='#95E1D3')),
        row=2, col=2
    )
    
    fig_multi.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig_multi, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de crecimiento
    col1, col2, col3 = st.columns(3)
    
    if len(annual) > 1:
        first_year = annual.iloc[0]
        last_year = annual.iloc[-1]
        
        with col1:
            growth_games = ((last_year['num_juegos'] - first_year['num_juegos']) / first_year['num_juegos'] * 100)
            st.metric(
                " Crecimiento en lanzamientos",
                f"{growth_games:+.1f}%",
                delta=f"{int(last_year['num_juegos'] - first_year['num_juegos'])} juegos"
            )
        
        with col2:
            growth_price = ((last_year['precio_promedio'] - first_year['precio_promedio']) / first_year['precio_promedio'] * 100)
            st.metric(
                " Cambio en precio promedio",
                f"{growth_price:+.1f}%",
                delta=f"${last_year['precio_promedio'] - first_year['precio_promedio']:.2f}"
            )
        
        with col3:
            rating_change = last_year['valoracion_promedio'] - first_year['valoracion_promedio']
            st.metric(
                " Cambio en valoración",
                f"{rating_change:+.1f}%",
                delta="Mejora" if rating_change > 0 else "Descenso"
            )

# ==================================================
# TAB 4 - INSIGHTS AVANZADOS
# ==================================================
with tab4:
    st.header(" Insights y Análisis Avanzado")
    
    # Análisis de segmentos
    st.subheader(" Segmentación de Mercado")
    
    # Crear segmentos por precio y valoración
    filtered_copy = filtered.copy()
    filtered_copy['precio_categoria'] = pd.cut(
        filtered_copy['price'], 
        bins=[0, 10, 30, 60, float('inf')],
        labels=['Económico', 'Medio', 'Premium', 'Lujo']
    )
    filtered_copy['valoracion_categoria'] = pd.cut(
        filtered_copy['porcentaje_positive_total'] * 100,
        bins=[0, 50, 70, 85, 100],
        labels=['Bajo', 'Medio', 'Alto', 'Excelente']
    )
    
    # Matriz de segmentación
    segment_matrix = filtered_copy.groupby(['precio_categoria', 'valoracion_categoria']).size().reset_index(name='count')
    
    if not segment_matrix.empty:
        fig_segments = px.density_heatmap(
            segment_matrix,
            x='precio_categoria',
            y='valoracion_categoria',
            z='count',
            color_continuous_scale='Blues',
            template="plotly_dark",
            title="Distribución de Juegos por Segmento"
        )
        fig_segments.update_layout(height=400)
        st.plotly_chart(fig_segments, use_container_width=True)
    
    st.markdown("---")
    
    # Hallazgos clave
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Hallazgos Clave del Análisis")
        
        # Calcular insights automáticos
        high_rated = filtered[filtered['porcentaje_positive_total'] > 0.85]
        low_reviews = filtered[filtered['total_num_reviews'] < filtered['total_num_reviews'].quantile(0.25)]
        
        st.markdown(f"""
        ** Insights principales:**
        
        1. **Calidad vs Popularidad**: {len(high_rated)} juegos ({len(high_rated)/len(filtered)*100:.1f}%) tienen valoraciones superiores al 85%, 
           pero solo el {len(high_rated[high_rated['total_num_reviews'] > filtered['total_num_reviews'].median()])/len(high_rated)*100:.1f}% 
           tiene popularidad por encima de la mediana.
        
        2. **Relación Precio-Calidad**: La correlación entre precio y valoración es de {filtered[['price', 'porcentaje_positive_total']].corr().iloc[0,1]:.3f}, 
           indicando que el precio {'es' if abs(filtered[['price', 'porcentaje_positive_total']].corr().iloc[0,1]) > 0.3 else 'no es'} un predictor fuerte de calidad.
        
        3. **Juegos de Nicho**: {len(low_reviews[low_reviews['porcentaje_positive_total'] > 0.8])} juegos de nicho (baja popularidad) 
           mantienen excelente recepción crítica (>80%).
        
        4. **Engagement**: Los juegos con valoraciones superiores a 80% tienen un tiempo promedio de juego de 
           {filtered[filtered['porcentaje_positive_total'] > 0.8]['average_playtime_forever'].mean():.1f} horas, 
           {((filtered[filtered['porcentaje_positive_total'] > 0.8]['average_playtime_forever'].mean() / filtered['average_playtime_forever'].mean() - 1) * 100):.1f}% 
           {'más' if filtered[filtered['porcentaje_positive_total'] > 0.8]['average_playtime_forever'].mean() > filtered['average_playtime_forever'].mean() else 'menos'} 
           que el promedio general.
        
        5. **Tendencia de Mercado**: El año {int(year)} presenta {len(filtered)} lanzamientos con un precio promedio de ${filtered['price'].mean():.2f}.
        """)
    
    with col2:
        st.subheader(" Top Performers")
        
        # Identificar mejores juegos
        if 'name' in filtered.columns:
            # Mejor valorado
            best_rated = filtered.nlargest(1, 'porcentaje_positive_total').iloc[0]
            st.markdown(f"** Mejor Valorado:**")
            st.info(f"{best_rated['name'] if 'name' in filtered.columns else 'N/A'}\n\n{best_rated['porcentaje_positive_total']*100:.1f}% positive")
            
            # Más popular
            most_popular = filtered.nlargest(1, 'total_num_reviews').iloc[0]
            st.markdown(f"** Más Popular:**")
            st.info(f"{most_popular['name'] if 'name' in filtered.columns else 'N/A'}\n\n{int(most_popular['total_num_reviews']):,} reviews")
            
            # Mejor relación calidad-precio
            filtered_copy['value_score'] = filtered_copy['porcentaje_positive_total'] / (filtered_copy['price'] + 1)
            best_value = filtered_copy.nlargest(1, 'value_score').iloc[0]
            st.markdown(f"** Mejor Valor:**")
            st.success(f"{best_value['name'] if 'name' in filtered.columns else 'N/A'}\n\n${best_value['price']:.2f} - {best_value['porcentaje_positive_total']*100:.1f}%")
    
    st.markdown("---")
    
    # Tabla de datos filtrados
    st.subheader(" Datos Filtrados")
    
    display_cols = ['name', 'price', 'porcentaje_positive_total', 'total_num_reviews', 
                    'average_playtime_forever', 'required_age', 'release_year']
    display_cols = [col for col in display_cols if col in filtered.columns]
    
    if display_cols:
        display_df = filtered[display_cols].copy()
        if 'porcentaje_positive_total' in display_df.columns:
            display_df['porcentaje_positive_total'] = (display_df['porcentaje_positive_total'] * 100).round(1)
            display_df.rename(columns={'porcentaje_positive_total': 'valoracion_%'}, inplace=True)
        
        st.dataframe(
            display_df.head(20),
            use_container_width=True,
            hide_index=True
        )
        
# ==================================================
# TAB 5 - GALERÍA VISUAL
# ==================================================
with tab5:
    st.header("Galería Visual y Presentación")
    
    # Sección de imagen de portada
    st.subheader("Wordclouds")
    
    banner_image = st.file_uploader(
        "Selecciona imagen de banner",
        type=["png", "jpg", "jpeg", "webp"],
        key="banner_upload"
    )

    banner_image2 = st.file_uploader(
        "Selecciona imagen de banner",
        type=["png", "jpg", "jpeg", "webp"],
        key="banner_upload"
    )
    
    if banner_image:
        from PIL import Image
        img = Image.open(banner_image)
        st.image(img, use_container_width=True, caption="Banner del Dashboard")
    else:
        st.info(" Sube una imagen de banner para darle personalidad a tu dashboard")

    if banner_image2:
        from PIL import Image
        img = Image.open(banner_image2)
        st.image(img, use_container_width=True, caption="Banner del Dashboard")
    else:
        st.info(" Sube una imagen de banner para darle personalidad a tu dashboard")
    st.markdown("---")
    
   
    

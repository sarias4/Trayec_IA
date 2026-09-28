"""
Taller de Analítica de Datos con IA
-----------------------------------
La app tiene 4 pestañas que se recorren en orden:
  1 · Carga de datos          → subes un CSV o eliges uno del repositorio
  2 · EDA estándar            → calidad, estadísticas y gráficos básicos
  3 · Análisis de datos       → comparaciones (cuantitativo) y texto (cualitativo)
  4 · Análisis con IA         → la IA interpreta tus datos y resultados
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Taller IA - Analítica Docente",
    layout="wide",
    initial_sidebar_state="expanded",
)

CARPETA = Path(__file__).parent
MODELOS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# Orden lógico de variables ordinales (se usa para ordenar tablas y gráficos)
ORDEN_CATEGORIAS = {
    "participacion_clase": ["Nula", "Baja", "Media", "Alta"],
    "horario_estudio": ["Mañana", "Tarde", "Noche"],
}

# Palabras muy comunes que no aportan al conteo de palabras
STOPWORDS = set(
    """para como pero porque cuando sobre tambien también hasta donde desde todo todos otros otras
    antes algunos estos estas esos esas mucho muchos poco estar algo este esta esto entre cada muy
    fueron tuve tuvo tener tiene tenia había hubo siempre durante varias varios cual quien
    ellos ellas nosotros mismo misma eran sido estaba estuvo sentí siento""".split()
)

PLANTILLAS = {
    "Patrones en los comentarios": (
        "Lee los comentarios de texto libre y agrúpalos en 3 o 4 temas recurrentes. "
        "Indica qué temas aparecen más entre los estudiantes con nota baja (menor a 3.0) y si el "
        "problema parece de capacidad o de causas logísticas/externas."
    ),
    "Diferencias entre grupos": (
        "Con los resultados calculados, explica qué grupo o combinación de variables se sale del "
        "patrón general y propón 2 hipótesis que lo expliquen. Advierte si alguna lectura global "
        "podría engañar al desagregar por subgrupos (por ejemplo, una Paradoja de Simpson)."
    ),
    "Recomendación para la coordinación": (
        "Escribe un párrafo de máximo 80 palabras dirigido a la coordinación académica, con una "
        "recomendación concreta y accionable basada en los datos y en los resultados calculados."
    ),
    "Calidad de los datos": (
        "Revisa la muestra y el resumen: ¿qué problemas de calidad ves (vacíos, valores raros, "
        "formatos mezclados) y qué pasos de limpieza propones antes de analizar?"
    ),
    "Personalizado (escribe el tuyo)": "",
}


# =============================== Funciones de apoyo ===============================
def leer_csv(fuente):
    """Lee un CSV (bytes o ruta). Prueba UTF-8 y Latin-1, y el separador ';'."""
    def _abrir():
        return io.BytesIO(fuente) if isinstance(fuente, (bytes, bytearray)) else fuente

    df, codificacion = None, "utf-8"
    for codificacion in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(_abrir(), encoding=codificacion)
            break
        except UnicodeDecodeError:
            continue
    if df is not None and df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(_abrir(), sep=";", encoding=codificacion)
    return df


def clasificar_columnas(df):
    """Separa las columnas en: numéricas, mixtas (número con texto), ids, textos y categóricas."""
    tipos = {"numericas": [], "mixtas": [], "ids": [], "textos": [], "categoricas": []}
    for col in df.columns:
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            tipos["numericas"].append(col)
            continue
        como_texto = serie.dropna().astype(str)
        if como_texto.empty:
            tipos["categoricas"].append(col)
        elif pd.to_numeric(como_texto, errors="coerce").notna().mean() >= 0.9:
            tipos["mixtas"].append(col)
        elif serie.nunique() == serie.notna().sum() and como_texto.str.len().mean() < 20:
            tipos["ids"].append(col)
        elif como_texto.str.len().mean() >= 40:
            tipos["textos"].append(col)
        else:
            tipos["categoricas"].append(col)
    return tipos


def etiqueta_tipo(col, tipos):
    if col in tipos["ids"]:
        return "Identificador"
    if col in tipos["numericas"]:
        return "Número"
    if col in tipos["mixtas"]:
        return "Número con texto mezclado ⚠️"
    if col in tipos["textos"]:
        return "Texto libre"
    return "Categoría"


def limpieza_rapida(df):
    """Convierte a número lo que parece numérico y rellena vacíos numéricos con la mediana."""
    df = df.copy()
    tipos = clasificar_columnas(df)
    for col in tipos["mixtas"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def orden_de(col, serie):
    """Orden de categorías: el lógico si existe, si no el alfabético."""
    if col in ORDEN_CATEGORIAS:
        presentes = set(serie.dropna().unique())
        return [c for c in ORDEN_CATEGORIAS[col] if c in presentes]
    return sorted(serie.dropna().unique().tolist(), key=str)


def frecuencia_palabras(serie, top=15):
    conteo = {}
    for texto in serie.dropna().astype(str):
        for palabra in re.findall(r"[a-záéíóúñü]{4,}", texto.lower()):
            if palabra not in STOPWORDS:
                conteo[palabra] = conteo.get(palabra, 0) + 1
    ordenado = sorted(conteo.items(), key=lambda x: -x[1])[:top]
    return pd.DataFrame(ordenado, columns=["palabra", "frecuencia"])


def grafico_palabras(tabla, titulo):
    if tabla.empty:
        st.caption("Sin palabras para mostrar.")
        return
    fig = px.bar(tabla.sort_values("frecuencia"), x="frecuencia", y="palabra", orientation="h", title=titulo)
    st.plotly_chart(fig)


# ---------- Apoyo para las gráficas del EDA ----------
PALETA = px.colors.qualitative.Bold


def mostrar(fig, clave, alto=None):
    """Muestra una gráfica de Plotly con márgenes compactos."""
    if alto:
        fig.update_layout(height=alto)
    fig.update_layout(margin=dict(t=55, b=30, l=30, r=20))
    st.plotly_chart(fig, key=clave)


def en_cuadricula(figuras, prefijo, columnas=2):
    """Reparte varias gráficas en columnas."""
    cols = st.columns(columnas)
    for i, fig in enumerate(figuras):
        with cols[i % columnas]:
            mostrar(fig, f"{prefijo}_{i}", 330)


def indice_de(lista, preferido):
    return lista.index(preferido) if preferido in lista else 0


def resumen_categoricas(df, cat):
    filas = []
    for c in cat:
        vc = df[c].value_counts()
        filas.append({
            "columna": c,
            "categorías": int(df[c].nunique()),
            "más frecuente": str(vc.index[0]) if len(vc) else "",
            "veces": int(vc.iloc[0]) if len(vc) else 0,
            "% del total": round(vc.iloc[0] / len(df) * 100, 1) if len(vc) else 0.0,
        })
    return pd.DataFrame(filas)


def resumen_atipicos(df, num):
    """Valores atípicos (regla del rango intercuartil), sesgo y curtosis por variable numérica."""
    filas = []
    for c in num:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < bajo) | (s > alto)).sum())
        filas.append({
            "variable": c, "atípicos": n, "% atípicos": round(n / len(s) * 100, 1),
            "límite inferior": round(bajo, 2), "límite superior": round(alto, 2),
            "sesgo": round(s.skew(), 2), "curtosis": round(s.kurt(), 2),
        })
    return pd.DataFrame(filas)


def frecuencia_bigramas(serie, top=15):
    """Pares de palabras que aparecen seguidas."""
    conteo = {}
    for texto in serie.dropna().astype(str):
        palabras = [p for p in re.findall(r"[a-záéíóúñü]{3,}", texto.lower()) if p not in STOPWORDS]
        for a, b in zip(palabras, palabras[1:]):
            clave = f"{a} {b}"
            conteo[clave] = conteo.get(clave, 0) + 1
    ordenado = sorted(conteo.items(), key=lambda x: -x[1])[:top]
    return pd.DataFrame(ordenado, columns=["frase", "frecuencia"])


# ================================ Barra lateral =================================
st.title("Taller de Analítica de Datos con IA")
st.sidebar.header("🔑 Configuración de la IA")

api_key_input = st.sidebar.text_input(
    "Pega aquí tu API Key de Groq",
    type="password",
    placeholder="gsk_...",
    help="Se usa solo mientras esta pestaña está abierta. No queda guardada en el repositorio.",
)
api_key = api_key_input.strip()
if not api_key:
    try:  # Alternativa: llave guardada en Settings -> Secrets
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

if api_key:
    st.sidebar.success("Llave detectada ✅")
else:
    st.sidebar.warning("Falta la API Key. Pégala arriba para usar la pestaña 4.")

modelo_lista = st.sidebar.selectbox("Modelo de IA", MODELOS, index=0)
modelo_manual = st.sidebar.text_input(
    "¿Otro modelo? Escribe su ID (opcional)", placeholder="ej. openai/gpt-oss-120b"
)
modelo = modelo_manual.strip() or modelo_lista
temperatura = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.5, 0.4, 0.1)
max_filas = st.sidebar.slider("Filas de datos que se envían a la IA", 20, 200, 100, 10)

# ==================================== Pestañas ===================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["1 · Carga de datos", "2 · EDA estándar", "3 · Análisis de datos", "4 · Análisis con IA"]
)

df = None                 # DataFrame con el que trabajan las pestañas 2, 3 y 4
nombre_datos = ""
contexto_extra = ""       # Texto adicional que viaja a la IA (ver extensión en la pestaña 3)

# ------------------------------- 1 · Carga de datos -------------------------------
with tab1:
    st.subheader("Carga de datos")
    origen = st.radio(
        "¿De dónde quieres cargar los datos?",
        ["Subir un archivo CSV", "Usar un CSV del repositorio"],
        horizontal=True,
    )

    if origen == "Subir un archivo CSV":
        archivo = st.file_uploader("Sube tu CSV (basico.csv / medio.csv / experto.csv)", type="csv")
        if archivo is not None:
            try:
                st.session_state["df_original"] = leer_csv(archivo.getvalue())
                st.session_state["nombre_datos"] = archivo.name
            except Exception as e:
                st.error(f"No pude leer ese archivo: {e}")
    else:
        csvs = sorted(p.name for p in CARPETA.glob("*.csv"))
        if csvs:
            elegido = st.selectbox("CSV encontrados en el repositorio", csvs)
            if st.button("Cargar este archivo", key="btn_cargar_repo"):
                try:
                    st.session_state["df_original"] = leer_csv(CARPETA / elegido)
                    st.session_state["nombre_datos"] = elegido
                except Exception as e:
                    st.error(f"No pude leer ese archivo: {e}")
        else:
            st.warning("No hay archivos .csv en la carpeta del repositorio. Sube uno con la otra opción.")

    df_original = st.session_state.get("df_original")
    if df_original is None:
        st.info("Aún no hay datos cargados. Sube un archivo o elige uno del repositorio.")
    else:
        nombre_datos = st.session_state.get("nombre_datos", "datos")
        limpiar = st.checkbox(
            "Limpieza rápida (convierte a número lo que parezca numérico y rellena vacíos numéricos con la mediana)",
            value=False,
        )
        df = limpieza_rapida(df_original) if limpiar else df_original

        tipos = clasificar_columnas(df)
        st.success(f"Datos cargados: **{nombre_datos}** — {df.shape[0]} filas × {df.shape[1]} columnas")
        c1, c2, c3 = st.columns(3)
        c1.metric("Columnas numéricas", len(tipos["numericas"]))
        c2.metric("Columnas categóricas", len(tipos["categoricas"]))
        c3.metric("Columnas de texto libre", len(tipos["textos"]))
        st.markdown("**Vista previa (primeras 20 filas):**")
        st.dataframe(df.head(20))

# --------------------------------- 2 · EDA estándar --------------------------------
with tab2:
    st.subheader("Exploración estándar de los datos (EDA)")
    if df is None:
        st.info("Primero carga tus datos en la pestaña 1.")
    else:
        tipos = clasificar_columnas(df)
        num, cat, textos_eda = tipos["numericas"], tipos["categoricas"], tipos["textos"]
        cat_graf = [c for c in cat if df[c].nunique() <= 25]   # categorías graficables

        e_res, e_num, e_cat, e_rel, e_txt, e_cal = st.tabs(
            ["📋 Resumen", "🔢 Numéricas", "🏷️ Categóricas", "🔗 Relaciones", "💬 Texto", "🧹 Calidad"]
        )

        # ============================ 📋 Resumen ============================
        with e_res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Filas", df.shape[0])
            c2.metric("Columnas", df.shape[1])
            c3.metric("Filas duplicadas", int(df.duplicated().sum()))
            c4.metric("Celdas vacías", int(df.isna().sum().sum()))

            g1, g2 = st.columns(2)
            with g1:
                composicion = pd.Series([etiqueta_tipo(c, tipos) for c in df.columns]).value_counts().reset_index()
                composicion.columns = ["tipo", "columnas"]
                mostrar(px.pie(composicion, names="tipo", values="columnas", hole=0.5,
                               title="Tipos de columnas", color_discrete_sequence=PALETA), "eda_tipos", 340)
            with g2:
                completo = ((1 - df.isna().mean()) * 100).round(1).reset_index()
                completo.columns = ["columna", "% completo"]
                mostrar(px.bar(completo.sort_values("% completo"), x="% completo", y="columna", orientation="h",
                               range_x=[0, 100], title="% de datos completos por columna",
                               color_discrete_sequence=["#2A9D8F"]), "eda_completo", 340)

            if num:
                st.markdown("### Estadísticas de las variables numéricas")
                st.dataframe(df[num].describe().T.round(2))
            if cat:
                st.markdown("### Resumen de las variables categóricas")
                st.dataframe(resumen_categoricas(df, cat))

        # ============================ 🔢 Numéricas ============================
        with e_num:
            if not num:
                st.info("No hay columnas numéricas en estos datos. Si hay números guardados como texto, "
                        "usa la 'Limpieza rápida' de la pestaña 1.")
            else:
                st.markdown("### Distribución de cada variable numérica")
                figs = [
                    px.histogram(df, x=c, marginal="box", nbins=20, title=c,
                                 color_discrete_sequence=[PALETA[i % len(PALETA)]])
                    for i, c in enumerate(num[:8])
                ]
                en_cuadricula(figs, "eda_hist")
                if len(num) > 8:
                    st.caption(f"Se muestran las primeras 8 de {len(num)} variables numéricas.")

                st.markdown("### Explora una variable en detalle")
                v = st.selectbox("Variable", num, index=indice_de(num, "nota_final"), key="eda_num_var")
                bins = st.slider("Número de barras del histograma", 5, 60, 20, key="eda_bins")
                serie = df[v].dropna()
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Promedio", f"{serie.mean():.2f}")
                m2.metric("Mediana", f"{serie.median():.2f}")
                m3.metric("Desviación", f"{serie.std():.2f}")
                m4.metric("Sesgo", f"{serie.skew():.2f}")
                m5.metric("Curtosis", f"{serie.kurt():.2f}")
                a, b = st.columns(2)
                with a:
                    mostrar(px.histogram(df, x=v, nbins=bins, marginal="violin", histnorm="probability density",
                                         title=f"Densidad de {v}", color_discrete_sequence=["#264653"]),
                            "eda_detalle_hist", 380)
                with b:
                    mostrar(px.ecdf(df, x=v, title=f"Distribución acumulada (ECDF) de {v}",
                                    color_discrete_sequence=["#E76F51"]), "eda_detalle_ecdf", 380)

                st.markdown("### Comparación entre variables numéricas")
                atipicos = resumen_atipicos(df, num)
                z = (df[num] - df[num].mean()) / df[num].std(ddof=0).replace(0, float("nan"))
                z = z.melt(var_name="variable", value_name="valor estandarizado")
                k1, k2 = st.columns(2)
                with k1:
                    mostrar(px.box(z, x="variable", y="valor estandarizado", color="variable", points="outliers",
                                   title="Cajas comparadas (valores estandarizados)",
                                   color_discrete_sequence=PALETA).update_layout(showlegend=False),
                            "eda_cajas_z", 380)
                with k2:
                    mostrar(px.bar(atipicos, x="variable", y="% atípicos", text="atípicos", color="variable",
                                   title="Valores atípicos por variable (regla del rango intercuartil)",
                                   color_discrete_sequence=PALETA).update_layout(showlegend=False),
                            "eda_atipicos", 380)
                k3, k4 = st.columns(2)
                with k3:
                    mostrar(px.bar(atipicos, x="variable", y="sesgo", color="sesgo",
                                   color_continuous_scale="RdBu_r", range_color=[-2, 2],
                                   title="Sesgo de cada variable (0 = simétrica)"), "eda_sesgo", 340)
                with k4:
                    medias = df[num].mean().reset_index()
                    medias.columns = ["variable", "promedio"]
                    mostrar(px.bar(medias, x="variable", y="promedio", text_auto=".2f", color="variable",
                                   title="Promedio de cada variable",
                                   color_discrete_sequence=PALETA).update_layout(showlegend=False),
                            "eda_medias", 340)
                st.dataframe(atipicos)

        # ============================ 🏷️ Categóricas ============================
        with e_cat:
            if not cat_graf:
                st.info("No hay variables categóricas graficables en estos datos.")
            else:
                st.markdown("### Frecuencia de cada variable categórica")
                figs = []
                for c in cat_graf[:6]:
                    conteo = df[c].value_counts().reset_index()
                    conteo.columns = [c, "cantidad"]
                    fig = px.bar(conteo, x=c, y="cantidad", text="cantidad", color=c, title=c,
                                 color_discrete_sequence=PALETA, category_orders={c: orden_de(c, df[c])})
                    figs.append(fig.update_layout(showlegend=False))
                en_cuadricula(figs, "eda_cat")

                st.markdown("### Proporciones de una variable")
                cv = st.selectbox("Variable categórica", cat_graf, key="eda_cat_var")
                conteo = df[cv].value_counts().reset_index()
                conteo.columns = [cv, "cantidad"]
                d1, d2 = st.columns(2)
                with d1:
                    mostrar(px.pie(conteo, names=cv, values="cantidad", hole=0.45,
                                   title=f"Proporción de {cv}", color_discrete_sequence=PALETA), "eda_pie", 360)
                with d2:
                    mostrar(px.treemap(conteo, path=[cv], values="cantidad",
                                       title=f"Mapa de árbol de {cv}", color_discrete_sequence=PALETA),
                            "eda_treemap", 360)

                if len(cat_graf) >= 2:
                    st.markdown("### Relación entre dos categorías")
                    f1, f2 = st.columns(2)
                    ca = f1.selectbox("Primera categoría", cat_graf, index=0, key="eda_cat_a")
                    cb = f2.selectbox("Segunda categoría", cat_graf, index=1, key="eda_cat_b")
                    if ca != cb:
                        cruce = pd.crosstab(df[ca], df[cb])
                        cruce = cruce.reindex(index=orden_de(ca, df[ca]), columns=orden_de(cb, df[cb]))
                        h1, h2 = st.columns(2)
                        with h1:
                            mostrar(px.imshow(cruce, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                                              title=f"Conteo: {ca} × {cb}"), "eda_cruce_conteo", 380)
                        with h2:
                            mostrar(px.histogram(df, x=ca, color=cb, barnorm="percent", text_auto=".0f",
                                                 title=f"Composición de {cb} dentro de cada {ca} (%)",
                                                 color_discrete_sequence=PALETA,
                                                 category_orders={ca: orden_de(ca, df[ca]), cb: orden_de(cb, df[cb])}),
                                    "eda_cruce_compo", 380)
                        pares = df.groupby([ca, cb]).size().reset_index(name="cantidad")
                        mostrar(px.sunburst(pares, path=[ca, cb], values="cantidad",
                                            title=f"Sunburst: {ca} → {cb}", color_discrete_sequence=PALETA),
                                "eda_sunburst", 420)
                    else:
                        st.caption("Elige dos categorías distintas.")

                    st.markdown("### Flujo entre categorías (categorías paralelas)")
                    dims = cat_graf[:4]
                    color_par = num[indice_de(num, "nota_final")] if num else None
                    fig_par = px.parallel_categories(df, dimensions=dims, color=color_par,
                                                     color_continuous_scale="Viridis",
                                                     title="Cada línea es un grupo de filas; el color indica el promedio" if color_par else None)
                    mostrar(fig_par, "eda_paralelas", 420)

        # ============================ 🔗 Relaciones ============================
        with e_rel:
            if len(num) >= 2:
                metodo = st.radio(
                    "Tipo de correlación", ["pearson", "spearman"], horizontal=True, key="eda_metodo",
                    format_func=lambda m: "Pearson (lineal)" if m == "pearson" else "Spearman (por rangos)",
                )
                corr = df[num].corr(method=metodo).round(2)
                pares_corr = pd.DataFrame(
                    [(f"{a} × {b}", corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:]],
                    columns=["par de variables", "correlación"],
                )
                pares_corr["fuerza"] = pares_corr["correlación"].abs()
                top = pares_corr.sort_values("fuerza", ascending=False).head(10).sort_values("fuerza")
                r1, r2 = st.columns(2)
                with r1:
                    mostrar(px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                      aspect="auto", title="Mapa de calor de correlaciones"), "eda_corr_mapa", 420)
                with r2:
                    mostrar(px.bar(top, x="correlación", y="par de variables", orientation="h", color="correlación",
                                   color_continuous_scale="RdBu_r", range_color=[-1, 1],
                                   title="Pares de variables más relacionados"), "eda_corr_top", 420)

                st.markdown("### Matriz de dispersión")
                colorear = st.selectbox("Colorear puntos por", ["(ninguno)"] + cat_graf, key="eda_matriz_color")
                fig_m = px.scatter_matrix(df, dimensions=num[:5], color=None if colorear == "(ninguno)" else colorear,
                                          color_discrete_sequence=PALETA, height=620)
                fig_m.update_traces(diagonal_visible=False, showupperhalf=False,
                                    marker=dict(size=4, opacity=0.7))
                mostrar(fig_m, "eda_matriz", 620)

                st.markdown("### Dos variables frente a frente")
                s1, s2, s3 = st.columns(3)
                x = s1.selectbox("Eje X", num, index=0, key="eda_x")
                y = s2.selectbox("Eje Y", num, index=1, key="eda_y")
                col_sc = s3.selectbox("Color", ["(ninguno)"] + cat_graf, key="eda_color_sc")
                fig_sc = px.scatter(df, x=x, y=y, color=None if col_sc == "(ninguno)" else col_sc,
                                    marginal_x="histogram", marginal_y="box", opacity=0.75,
                                    color_discrete_sequence=PALETA, title=f"{y} según {x}")
                datos_xy = df[[x, y]].dropna()
                if x != y and len(datos_xy) > 2 and datos_xy[x].nunique() > 1:
                    pendiente, intercepto = np.polyfit(datos_xy[x], datos_xy[y], 1)
                    xs = np.array([datos_xy[x].min(), datos_xy[x].max()])
                    fig_sc.add_trace(go.Scatter(x=xs, y=pendiente * xs + intercepto, mode="lines",
                                                name="Tendencia lineal", line=dict(color="black", dash="dash")),
                                     row=1, col=1)
                    st.caption(f"Correlación entre {x} y {y}: {datos_xy[x].corr(datos_xy[y]):.2f} · "
                               f"por cada unidad de {x}, {y} cambia en promedio {pendiente:.2f}.")
                mostrar(fig_sc, "eda_scatter", 480)
            else:
                st.info("Necesitas al menos 2 variables numéricas para ver correlaciones y dispersión.")

            if num and cat_graf:
                st.markdown("### Una variable numérica según una categoría")
                n1, n2 = st.columns(2)
                vn = n1.selectbox("Variable numérica", num, index=indice_de(num, "nota_final"), key="eda_vn")
                vc = n2.selectbox("Categoría", cat_graf, key="eda_vc")
                orden = {vc: orden_de(vc, df[vc])}
                p1, p2 = st.columns(2)
                with p1:
                    mostrar(px.violin(df, x=vc, y=vn, color=vc, box=True, points="all", category_orders=orden,
                                      color_discrete_sequence=PALETA, title=f"{vn} por {vc} (violín)"
                                      ).update_layout(showlegend=False), "eda_violin", 400)
                with p2:
                    resumen_g = df.groupby(vc)[vn].agg(promedio="mean", desviacion="std").reset_index()
                    mostrar(px.bar(resumen_g, x=vc, y="promedio", error_y="desviacion", color=vc,
                                   category_orders=orden, color_discrete_sequence=PALETA,
                                   title=f"Promedio de {vn} por {vc} (± desviación)"
                                   ).update_layout(showlegend=False), "eda_media_error", 400)
                mostrar(px.histogram(df, x=vn, color=vc, barmode="overlay", opacity=0.6, marginal="box",
                                     category_orders=orden, color_discrete_sequence=PALETA,
                                     title=f"Distribuciones de {vn} superpuestas por {vc}"), "eda_overlay", 420)

            if len(num) >= 3:
                with st.expander("Ver coordenadas paralelas (todas las numéricas a la vez)"):
                    color_p = st.selectbox("Colorear por", num, index=indice_de(num, "nota_final"), key="eda_par_color")
                    mostrar(px.parallel_coordinates(df[num].dropna(), dimensions=num, color=color_p,
                                                    color_continuous_scale="Viridis"), "eda_coord_par", 420)

        # ============================ 💬 Texto ============================
        with e_txt:
            if not textos_eda:
                st.info("No detecté columnas de texto libre (por ejemplo, comentarios). "
                        "Usa basico.csv, experto.csv o maestro.csv para esta sección.")
            else:
                ct = st.selectbox("Columna de texto", textos_eda, key="eda_txt_col")
                txt = df[ct].fillna("").astype(str)
                largo_car, largo_pal = txt.str.len(), txt.str.split().str.len()
                t1, t2, t3 = st.columns(3)
                t1.metric("Textos", int((txt != "").sum()))
                t2.metric("Textos distintos", int(txt.nunique()))
                t3.metric("Palabras por texto (promedio)", f"{largo_pal.mean():.1f}")

                a, b = st.columns(2)
                with a:
                    mostrar(px.histogram(largo_car, nbins=20, title="Longitud en caracteres",
                                         labels={"value": "caracteres"},
                                         color_discrete_sequence=["#2A9D8F"]).update_layout(showlegend=False),
                            "eda_txt_car", 340)
                with b:
                    mostrar(px.histogram(largo_pal, nbins=15, title="Cantidad de palabras",
                                         labels={"value": "palabras"},
                                         color_discrete_sequence=["#E9C46A"]).update_layout(showlegend=False),
                            "eda_txt_pal", 340)

                st.markdown("### Palabras y frases más frecuentes")
                w1, w2 = st.columns(2)
                with w1:
                    tabla_p = frecuencia_palabras(df[ct], top=15)
                    if not tabla_p.empty:
                        mostrar(px.bar(tabla_p.sort_values("frecuencia"), x="frecuencia", y="palabra",
                                       orientation="h", title="Top 15 palabras",
                                       color_discrete_sequence=["#264653"]), "eda_txt_palabras", 440)
                with w2:
                    tabla_b = frecuencia_bigramas(df[ct], top=15)
                    if not tabla_b.empty:
                        mostrar(px.bar(tabla_b.sort_values("frecuencia"), x="frecuencia", y="frase",
                                       orientation="h", title="Top 15 pares de palabras seguidas",
                                       color_discrete_sequence=["#E76F51"]), "eda_txt_bigramas", 440)

                st.markdown("### Textos más repetidos")
                repetidos = df[ct].value_counts().head(10).reset_index()
                repetidos.columns = ["texto", "veces"]
                repetidos["texto"] = repetidos["texto"].apply(lambda t: t if len(t) <= 60 else t[:60] + "…")
                mostrar(px.bar(repetidos.sort_values("veces"), x="veces", y="texto", orientation="h",
                               title="Los 10 textos que más se repiten", color_discrete_sequence=["#8AB17D"]),
                        "eda_txt_repetidos", 420)

                if num:
                    st.markdown("### ¿La longitud del texto se relaciona con una variable numérica?")
                    vt = st.selectbox("Variable numérica", num, index=indice_de(num, "nota_final"), key="eda_txt_num")
                    datos_t = pd.DataFrame({"palabras": largo_pal, vt: df[vt]})
                    mostrar(px.scatter(datos_t, x="palabras", y=vt, opacity=0.7,
                                       title=f"{vt} según la cantidad de palabras del texto",
                                       color_discrete_sequence=["#6A4C93"]), "eda_txt_scatter", 380)

        # ============================ 🧹 Calidad ============================
        with e_cal:
            calidad = pd.DataFrame(
                {
                    "columna": df.columns,
                    "tipo detectado": [etiqueta_tipo(c, tipos) for c in df.columns],
                    "vacíos": df.isna().sum().values,
                    "% vacíos": (df.isna().mean() * 100).round(1).values,
                    "valores únicos": df.nunique().values,
                }
            )
            st.dataframe(calidad)
            for col in tipos["mixtas"]:
                no_numerico = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
                raros = df.loc[no_numerico, col].astype(str).unique().tolist()
                st.warning(
                    f"⚠️ La columna **{col}** parece numérica pero tiene texto mezclado: {raros[:5]}. "
                    "Puedes corregirlo con la 'Limpieza rápida' de la pestaña 1."
                )
            constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            if constantes:
                st.warning(f"Columnas con un solo valor (no aportan información): {constantes}")
            if int(df.duplicated().sum()) > 0:
                st.warning(f"Hay {int(df.duplicated().sum())} filas duplicadas.")

            hay_vacios = calidad[calidad["vacíos"] > 0]
            if hay_vacios.empty:
                st.success("No hay celdas vacías en estos datos ✅")
            else:
                q1, q2 = st.columns(2)
                with q1:
                    mostrar(px.bar(hay_vacios, x="columna", y="vacíos", text="vacíos",
                                   title="Celdas vacías por columna", color_discrete_sequence=["#D62728"]),
                            "eda_vacios_barra", 380)
                with q2:
                    mostrar(px.imshow(df.head(300).isna().astype(int).T, aspect="auto",
                                      color_continuous_scale=[[0, "#EEEEEE"], [1, "#D62728"]],
                                      title="Mapa de celdas vacías (rojo = vacío)"
                                      ).update_coloraxes(showscale=False), "eda_vacios_mapa", 380)

# ----------------------- 3 · Análisis cuantitativo y cualitativo ------------------------
resumen_analisis = ""
with tab3:
    st.subheader("Análisis de datos: cuantitativo y cualitativo")
    if df is None:
        st.info("Primero carga tus datos en la pestaña 1.")
    else:
        tipos = clasificar_columnas(df)
        num, cat, textos = tipos["numericas"], tipos["categoricas"], tipos["textos"]
        partes = [f"Archivo: {nombre_datos} ({df.shape[0]} filas, columnas: {', '.join(map(str, df.columns))})"]

        # ---------- Cuantitativo ----------
        st.markdown("## 📊 Análisis cuantitativo")
        objetivo = None
        if not num:
            st.warning("No hay columnas numéricas para analizar. Usa la 'Limpieza rápida' de la pestaña 1 "
                       "si hay números guardados como texto.")
        else:
            idx = num.index("nota_final") if "nota_final" in num else 0
            objetivo = st.selectbox("Variable numérica a analizar", num, index=idx, key="objetivo")
            serie_obj = df[objetivo].dropna()
            partes.append(
                f"Variable analizada: {objetivo}. Promedio {serie_obj.mean():.2f}, mediana {serie_obj.median():.2f}, "
                f"mínimo {serie_obj.min():.2f}, máximo {serie_obj.max():.2f}, desviación {serie_obj.std():.2f}."
            )

            if cat:
                grupos = st.multiselect(
                    f"Compara '{objetivo}' según estas categorías (máximo 3)",
                    cat, default=cat[:1], max_selections=3, key="grupos",
                )
                if grupos:
                    tabla = (
                        df.groupby(grupos, observed=True)[objetivo]
                        .agg(cantidad="count", promedio="mean", mediana="median", desviacion="std")
                        .round(2)
                        .reset_index()
                    )
                    st.dataframe(tabla)
                    ordenes = {g: orden_de(g, df[g]) for g in grupos}
                    color = grupos[1] if len(grupos) > 1 else None
                    st.plotly_chart(
                        px.bar(tabla, x=grupos[0], y="promedio", color=color, barmode="group",
                               title=f"Promedio de {objetivo} por {' y '.join(grupos)}",
                               category_orders=ordenes)
                    )
                    st.plotly_chart(
                        px.box(df, x=grupos[0], y=objetivo, color=color,
                               title=f"Distribución de {objetivo} por {' y '.join(grupos)}",
                               category_orders=ordenes)
                    )
                    partes.append(f"Promedio de {objetivo} por {' y '.join(grupos)}:\n{tabla.to_string(index=False)}")

            if len(cat) >= 2:
                st.markdown("**Cruce de dos categorías (mapa de calor del promedio)**")
                f1, f2 = st.columns(2)
                fila = f1.selectbox("Categoría en las filas", cat, index=0, key="cruce_fila")
                columna = f2.selectbox("Categoría en las columnas", cat, index=1, key="cruce_columna")
                if fila != columna:
                    pivote = df.pivot_table(index=fila, columns=columna, values=objetivo, aggfunc="mean").round(2)
                    pivote = pivote.reindex(index=orden_de(fila, df[fila]), columns=orden_de(columna, df[columna]))
                    st.plotly_chart(
                        px.imshow(pivote, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
                                  title=f"Promedio de {objetivo}: {fila} × {columna}")
                    )
                    partes.append(f"Cruce {fila} × {columna} (promedio de {objetivo}):\n{pivote.to_string()}")
                else:
                    st.caption("Elige dos categorías distintas para ver el cruce.")

            otras = [c for c in num if c != objetivo]
            if otras:
                correl = df[otras + [objetivo]].corr()[objetivo].drop(objetivo).round(2).sort_values()
                st.markdown(f"**Correlación de las demás variables numéricas con {objetivo}**")
                tabla_corr = correl.reset_index()
                tabla_corr.columns = ["variable", "correlación"]
                st.plotly_chart(px.bar(tabla_corr, x="correlación", y="variable", orientation="h"))
                partes.append(f"Correlación con {objetivo}:\n{correl.to_string()}")

        # ---------- Cualitativo ----------
        st.markdown("## 💬 Análisis cualitativo (texto libre)")
        if not textos:
            st.info("No detecté columnas de texto libre en estos datos (por ejemplo, comentarios). "
                    "Usa el archivo basico.csv o experto.csv para esta sección.")
        else:
            col_texto = st.selectbox("Columna de texto a analizar", textos, key="col_texto")
            st.markdown("**Palabras más frecuentes**")
            frec = frecuencia_palabras(df[col_texto], top=15)
            grafico_palabras(frec, "Top 15 palabras")
            if not frec.empty:
                partes.append("Palabras más frecuentes en los comentarios: " + ", ".join(
                    f"{r.palabra} ({r.frecuencia})" for r in frec.head(10).itertuples()))

            st.markdown("**¿Qué temas se mencionan y cómo se relacionan con la nota?**")
            claves_txt = st.text_input(
                "Palabras clave (separadas por coma; basta con el inicio de la palabra)",
                value="internet, conexi, conect, trabajo, tiempo, coordin, computador, turnos, señal, plataforma",
                key="claves",
            )
            claves = [k.strip().lower() for k in claves_txt.split(",") if k.strip()]
            textos_min = df[col_texto].fillna("").astype(str).str.lower()
            if claves:
                menciona = textos_min.apply(lambda t: any(k in t for k in claves))
            else:
                menciona = pd.Series(False, index=df.index)
            st.metric("Comentarios que mencionan alguna palabra clave", f"{int(menciona.sum())} de {len(df)}")

            if objetivo is not None:
                minimo, maximo = float(df[objetivo].min()), float(df[objetivo].max())
                if minimo < maximo:
                    umbral = st.slider(
                        f"Umbral de {objetivo} para separar 'bajo' de 'alto'",
                        minimo, maximo, min(max(3.0, minimo), maximo), key="umbral",
                    )
                    bajo = df[objetivo] < umbral
                    tabla_m = (
                        pd.DataFrame({
                            "menciona": menciona.map({True: "Menciona palabra clave", False: "No la menciona"}),
                            objetivo: df[objetivo],
                        })
                        .groupby("menciona")[objetivo].agg(cantidad="count", promedio="mean").round(2).reset_index()
                    )
                    st.dataframe(tabla_m)
                    if bajo.sum() > 0 and (~bajo).sum() > 0:
                        pct_bajo, pct_alto = menciona[bajo].mean() * 100, menciona[~bajo].mean() * 100
                        a, b = st.columns(2)
                        a.metric(f"% que menciona el tema ({objetivo} < {umbral:.1f})", f"{pct_bajo:.0f}%")
                        b.metric(f"% que menciona el tema ({objetivo} ≥ {umbral:.1f})", f"{pct_alto:.0f}%")
                        partes.append(
                            f"Palabras clave usadas: {', '.join(claves)}. Entre los que tienen {objetivo} < {umbral:.1f}, "
                            f"{pct_bajo:.0f}% menciona alguna; entre los que tienen {objetivo} >= {umbral:.1f}, {pct_alto:.0f}%.\n"
                            f"Promedio de {objetivo} según mencione o no el tema:\n{tabla_m.to_string(index=False)}"
                        )
                        w1, w2 = st.columns(2)
                        with w1:
                            grafico_palabras(frecuencia_palabras(df.loc[bajo, col_texto], 10), f"Palabras: {objetivo} bajo")
                        with w2:
                            grafico_palabras(frecuencia_palabras(df.loc[~bajo, col_texto], 10), f"Palabras: {objetivo} alto")
            with st.expander("Ver comentarios que mencionan una palabra clave"):
                cols_ver = ([objetivo] if objetivo else []) + [col_texto]
                st.dataframe(df.loc[menciona, cols_ver].head(30))

        # EXTENSIÓN (Integrador / Experto): agrega aquí tu propio análisis.
        # Todo lo que sumes a `contexto_extra` (contexto_extra += "texto") viaja a la IA en la pestaña 4.

        if contexto_extra:
            partes.append("Análisis adicional:\n" + contexto_extra)
        resumen_analisis = "\n\n".join(partes)[:7000]

# --------------------------------- 4 · Análisis con IA --------------------------------
with tab4:
    st.subheader("Análisis con IA")
    if df is None:
        st.info("Primero carga tus datos en la pestaña 1.")
    else:
        st.caption(f"Modelo: **{modelo}** · Creatividad: {temperatura} · Configúralo en el menú de la izquierda 👈")
        preset = st.selectbox("Elige una plantilla o escribe la tuya", list(PLANTILLAS.keys()))
        instruccion = st.text_area("Instrucción para la IA", value=PLANTILLAS[preset], height=150, key=f"prompt_{preset}")

        o1, o2 = st.columns(2)
        incluir_resumen = o1.checkbox("Incluir el resumen de la pestaña 3", value=True)
        incluir_datos = o2.checkbox(f"Incluir una muestra de los datos ({max_filas} filas)", value=True)

        partes_msg = [instruccion.strip()]
        if incluir_resumen and resumen_analisis:
            partes_msg.append("Resultados calculados en la app:\n" + resumen_analisis)
        if incluir_datos:
            partes_msg.append(f"Datos (CSV, primeras {max_filas} filas):\n" + df.head(max_filas).to_csv(index=False))
        mensaje = "\n\n".join(p for p in partes_msg if p)

        tokens = len(mensaje) // 4
        st.caption(f"Tamaño aproximado del mensaje: ~{tokens:,} tokens")
        if tokens > 6000:
            st.warning("El mensaje es grande. Si aparece un error de límite (413/429), baja las filas en el menú "
                       "izquierdo o desmarca la muestra de datos.")

        if st.button("Analizar con IA", type="primary", key="btn_ia"):
            if not api_key:
                st.error("Falta tu API Key de Groq. Pégala en el menú de la izquierda 👈")
            elif not instruccion.strip():
                st.warning("Escribe primero una instrucción para la IA.")
            else:
                try:
                    cliente = Groq(api_key=api_key)
                    with st.spinner("La IA está analizando..."):
                        respuesta = cliente.chat.completions.create(
                            model=modelo,
                            temperature=temperatura,
                            messages=[
                                {"role": "system",
                                 "content": "Eres un analista de datos educativos. Responde en español, de forma clara y concisa."},
                                {"role": "user", "content": mensaje},
                            ],
                        )
                    texto = respuesta.choices[0].message.content
                    st.session_state["respuesta_ia"] = texto or "(La IA no devolvió texto. Prueba de nuevo o cambia de modelo.)"
                    st.session_state["mensaje_enviado"] = mensaje
                except Exception as e:
                    st.error(f"No se pudo obtener respuesta de la IA: {e}")
                    st.info(
                        "Si el error menciona límite (429), espera 30 segundos y reintenta. "
                        "Si menciona tamaño (413), baja las filas en el menú izquierdo. "
                        "Si dice que el modelo no existe, cambia el modelo en el menú izquierdo."
                    )

        if st.session_state.get("respuesta_ia"):
            st.markdown("### Respuesta de la IA")
            st.markdown(st.session_state["respuesta_ia"])
            st.download_button("Descargar respuesta (.md)", st.session_state["respuesta_ia"],
                               file_name="respuesta_ia.md", key="descarga_ia")
            with st.expander("Ver el mensaje completo que se envió a la IA"):
                st.text(st.session_state.get("mensaje_enviado", ""))

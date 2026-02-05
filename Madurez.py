import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Dashboard Madurez Digital", layout="wide")

EXCEL_PATH = "Madurez Digital, Tecnología y Analítica.xlsx"
SHEET_NAME = "DATA"

# Paleta (más “viva” y entendible)
PALETTE = px.colors.qualitative.Set2  # colores suaves pero variados


# =====================================================
# HELPERS
# =====================================================
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def yes_no(x):
    t = norm_text(x)
    if t.startswith("si") or t.startswith("sí"):
        return 1
    if t.startswith("no"):
        return 0
    return np.nan


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")


def madurez_level(score_0_100: float) -> str:
    if pd.isna(score_0_100):
        return "Sin dato"
    if score_0_100 < 50:
        return "🔴 Baja"
    if score_0_100 < 65:
        return "🟡 Media"
    if score_0_100 < 80:
        return "🟢 Buena"
    return "🟣 Alta"


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Columnas del Excel (tal como vienen en tu archivo)
    col_dep = '¿Qué tan “dependiente” es tu área de "datos" y "tecnología" para operar bien?'
    col_excel = "Nivel promedio de Excel"
    col_pbi = "¿Se usa Power BI en el área?"
    col_tools = "¿Conoces herramientas digitales? (Si tu respuesta es afirmativa, explicar cuales)"
    col_want_tools = "¿Te gustaría implementar alguna herramienta digital en tu área? (Si tu respuesta es afirmativa, explicar la herramienta)"
    col_dash = "Si usan dashboards: ¿qué tan confiables y útiles son para decisiones? (escala 1–5)"
    col_rep = "¿Qué tan seguido se crean reportes en tu área?"
    col_dataset = "En tu área existe un “dataset base” o archivo maestro que se use como fuente oficial."
    col_prob = "¿Qué tan seguido encuentran problemas con relación a datos?"
    col_dec = "En reuniones de seguimiento, ¿qué tanto se decide con datos vs “percepción”?"
    col_ideas = "¿Qué tan seguido el área propone ideas para mejorar con tecnología/herramientas digitales/datos?"
    col_ai_fam = "¿Qué tan familiarizo estás con herramientas de IA? 1-5"

    # Mapeos a escala 1–5
    map_dep = {"nada": 1, "poco": 2, "medio": 3, "mucho": 4, "total": 5}
    map_excel = {"básico": 2, "basico": 2, "intermedio": 3, "avanzado": 4, "experto": 5}
    map_pbi = {
        "no": 1,
        "sí, pero solo lo vemos/consultamos": 3,
        "si, pero solo lo vemos/consultamos": 3,
        "sí, y también creamos/modificamos reportes/tableros": 5,
        "si, y también creamos/modificamos reportes/tableros": 5,
    }
    map_rep = {"mensual": 2, "quincenal": 3, "semanal": 4, "diario": 5}
    # Menos problemas => mejor score
    map_prob = {"muy frecuente": 2, "frecuente": 3, "a veces": 4, "casi nunca": 5, "nunca": 5}
    map_dec = {
        "casi todo percepción": 1,
        "casi todo percepcion": 1,
        "mitad y mitad": 3,
        "mayormente con datos": 4,
        "siempre con datos": 5,
    }
    map_ideas = {"nunca": 1, "rara vez": 2, "a veces": 3, "frecuente": 4, "siempre": 5}

    # Scores base (0–5)
    d["score_dependencia"] = d[col_dep].map(lambda x: map_dep.get(norm_text(x), np.nan))
    d["score_excel"] = d[col_excel].map(lambda x: map_excel.get(norm_text(x), np.nan))
    d["score_powerbi"] = d[col_pbi].map(lambda x: map_pbi.get(norm_text(x), np.nan))
    d["score_dashboards"] = pd.to_numeric(d[col_dash], errors="coerce")
    d["score_freq_reportes"] = d[col_rep].map(lambda x: map_rep.get(norm_text(x), np.nan))
    d["score_dataset_base"] = d[col_dataset].map(yes_no).map(lambda v: 5 if v == 1 else (2 if v == 0 else np.nan))
    d["score_calidad_datos"] = d[col_prob].map(lambda x: map_prob.get(norm_text(x), np.nan))
    d["score_decisiones_con_datos"] = d[col_dec].map(lambda x: map_dec.get(norm_text(x), np.nan))
    d["score_ideas_mejora"] = d[col_ideas].map(lambda x: map_ideas.get(norm_text(x), np.nan))
    d["score_conoce_herr_digitales"] = d[col_tools].map(yes_no).map(lambda v: 4 if v == 1 else (2 if v == 0 else np.nan))
    d["score_ia_familiaridad"] = pd.to_numeric(d[col_ai_fam], errors="coerce")

    components = [
        "score_dependencia",
        "score_excel",
        "score_powerbi",
        "score_dashboards",
        "score_freq_reportes",
        "score_dataset_base",
        "score_calidad_datos",
        "score_decisiones_con_datos",
        "score_ideas_mejora",
        "score_conoce_herr_digitales",
        "score_ia_familiaridad",
    ]

    d["Madurez_0_5"] = d[components].mean(axis=1, skipna=True)
    d["Madurez_0_100"] = (d["Madurez_0_5"] / 5 * 100).round(1)
    d["Nivel_madurez"] = d["Madurez_0_100"].map(madurez_level)

    return d


# =====================================================
# LOAD
# =====================================================
try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(
        "❌ No encontré el Excel en la raíz del repo.\n\n"
        f"Verifica que exista: `{EXCEL_PATH}` y que el sheet se llame `{SHEET_NAME}`."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error leyendo el Excel con openpyxl:\n\n{e}")
    st.stop()

df = add_scores(df_raw)

# =====================================================
# SIDEBAR FILTERS (por defecto TODO)
# =====================================================
st.sidebar.title("🔎 Filtros (opcional)")
areas = sorted(df["Área"].dropna().unique().tolist()) if "Área" in df.columns else []
selected_areas = st.sidebar.multiselect("Área", options=areas, default=areas) if areas else []
show_people = st.sidebar.checkbox("Ver tabla por persona", value=True)

df_f = df[df["Área"].isin(selected_areas)].copy() if selected_areas else df.copy()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
<div style="padding: 14px 16px; border-radius: 14px; background: rgba(0,0,0,0.04);">
  <h2 style="margin:0;">📌 Resumen Ejecutivo</h2>
  <div style="margin-top:6px; color: rgba(255, 0, 0);">
    Este tablero muestra un índice de madurez (0–100) construido a partir de prácticas de datos, tecnología, BI, calidad de datos, cultura y familiaridad con IA.
    <br/>Por defecto ves <b>toda la organización</b>. 
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# =====================================================
# KPIs
# =====================================================
overall = float(df_f["Madurez_0_100"].mean()) if len(df_f) else np.nan
n_resp = int(len(df_f))

by_area = (
    df_f.groupby("Área", as_index=False)["Madurez_0_100"].mean()
    .sort_values("Madurez_0_100", ascending=False)
    if "Área" in df_f.columns and len(df_f)
    else pd.DataFrame(columns=["Área", "Madurez_0_100"])
)
best_area = by_area.iloc[0]["Área"] if len(by_area) else "-"
worst_area = by_area.iloc[-1]["Área"] if len(by_area) else "-"

pbi_col = "¿Se usa Power BI en el área?"
dataset_col = "En tu área existe un “dataset base” o archivo maestro que se use como fuente oficial."

pbi_yes = (
    df_f[pbi_col].astype(str).str.lower().str.startswith(("si", "sí")).mean() * 100
    if pbi_col in df_f.columns and len(df_f)
    else 0
)
dataset_yes = (
    df_f[dataset_col].astype(str).str.lower().str.startswith(("si", "sí")).mean() * 100
    if dataset_col in df_f.columns and len(df_f)
    else 0
)

c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])

with c1:
    # Gauge / indicador grande
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall if not np.isnan(overall) else 0,
            number={"suffix": " / 100"},
            title={"text": "Madurez promedio"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,0,0,0.25)"},
                    {"range": [50, 65], "color": "rgba(255,180,0,0.25)"},
                    {"range": [65, 80], "color": "rgba(0,200,0,0.20)"},
                    {"range": [80, 100], "color": "rgba(120,0,255,0.18)"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": overall if not np.isnan(overall) else 0},
            },
        )
    )
    fig_gauge.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c2:
    st.metric("Respuestas", f"{n_resp}")
    st.metric("Mediana", f"{df_f['Madurez_0_100'].median():.1f}" if len(df_f) else "-")

with c3:
    st.metric("Área más madura", f"{best_area}")
    st.metric("Área menos madura", f"{worst_area}")

with c4:
    st.metric("Power BI en uso", f"{pbi_yes:.0f}%")
    st.metric("Dataset base", f"{dataset_yes:.0f}%")

with c5:
    st.metric("Máx / Mín", f"{df_f['Madurez_0_100'].max():.1f} / {df_f['Madurez_0_100'].min():.1f}" if len(df_f) else "-")
    st.metric("Nivel global", madurez_level(overall) if not np.isnan(overall) else "-")

st.divider()

# =====================================================
# TABS (más entendible)
# =====================================================
tab1, tab2, tab3 = st.tabs(["📊 Por área", "🧩 Componentes", "📋 Detalle"])

# -------------------------
# TAB 1: Por área
# -------------------------
with tab1:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("🏢 Ranking de madurez por área (promedio)")
        if len(by_area):
            fig_area = px.bar(
                by_area,
                x="Madurez_0_100",
                y="Área",
                orientation="h",
                text="Madurez_0_100",
                color="Madurez_0_100",
                color_continuous_scale=["#ff4d4f", "#faad14", "#52c41a", "#722ed1"],
            )
            fig_area.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_area.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("No hay datos suficientes para graficar por área.")

    with right:
        st.subheader("📈 Distribución (personas)")
        fig_hist = px.histogram(df_f, x="Madurez_0_100", nbins=12, color_discrete_sequence=[PALETTE[0]])
        fig_hist.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("🚦 Semáforo de madurez (conteo)")
        order = ["🔴 Baja", "🟡 Media", "🟢 Buena", "🟣 Alta"]
        sem = df_f["Nivel_madurez"].value_counts().reindex(order).fillna(0).reset_index()
        sem.columns = ["Nivel", "Personas"]
        fig_sem = px.bar(
            sem,
            x="Nivel",
            y="Personas",
            text="Personas",
            color="Nivel",
            color_discrete_map={
                "🔴 Baja": "#ff4d4f",
                "🟡 Media": "#faad14",
                "🟢 Buena": "#52c41a",
                "🟣 Alta": "#722ed1",
            },
        )
        fig_sem.update_traces(textposition="outside")
        fig_sem.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sem, use_container_width=True)

    st.divider()

    st.subheader("🧠 Uso de BI y gobierno de datos (resumen rápido)")
    a1, a2, a3 = st.columns(3)

    if pbi_col in df_f.columns:
        pbi_counts = df_f[pbi_col].value_counts(dropna=False).reset_index()
        pbi_counts.columns = ["Respuesta", "Conteo"]
        fig_pbi = px.pie(pbi_counts, names="Respuesta", values="Conteo", hole=0.45, color_discrete_sequence=PALETTE)
        fig_pbi.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        a1.plotly_chart(fig_pbi, use_container_width=True)
        a1.caption("Power BI: tipo de uso (consulta vs creación).")

    if dataset_col in df_f.columns:
        ds = df_f[dataset_col].astype(str).str.strip().replace({"nan": "Sin respuesta"})
        ds_counts = ds.value_counts().reset_index()
        ds_counts.columns = ["Respuesta", "Conteo"]
        fig_ds = px.pie(ds_counts, names="Respuesta", values="Conteo", hole=0.45, color_discrete_sequence=PALETTE)
        fig_ds.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        a2.plotly_chart(fig_ds, use_container_width=True)
        a2.caption("Dataset base: existencia de fuente oficial por área.")

    col_prob = "¿Qué tan seguido encuentran problemas con relación a datos?"
    if col_prob in df_f.columns:
        prob_counts = df_f[col_prob].astype(str).value_counts().reset_index()
        prob_counts.columns = ["Frecuencia", "Conteo"]
        fig_prob = px.bar(prob_counts, x="Frecuencia", y="Conteo", text="Conteo", color_discrete_sequence=[PALETTE[1]])
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        a3.plotly_chart(fig_prob, use_container_width=True)
        a3.caption("Calidad de datos: frecuencia de problemas reportados.")

# -------------------------
# TAB 2: Componentes
# -------------------------
with tab2:
    st.subheader("🧩 Componentes que forman la madurez (promedio 0–5)")
    comp_cols = [
        ("Dependencia de datos/tecnología", "score_dependencia"),
        ("Nivel Excel", "score_excel"),
        ("Uso de Power BI", "score_powerbi"),
        ("Confiabilidad dashboards", "score_dashboards"),
        ("Frecuencia reportes", "score_freq_reportes"),
        ("Dataset base", "score_dataset_base"),
        ("Calidad de datos (menos problemas=mejor)", "score_calidad_datos"),
        ("Decisiones con datos", "score_decisiones_con_datos"),
        ("Ideas de mejora", "score_ideas_mejora"),
        ("Conoce herramientas digitales", "score_conoce_herr_digitales"),
        ("Familiaridad IA", "score_ia_familiaridad"),
    ]

    comp_df = pd.DataFrame(
        {
            "Componente": [a for a, _ in comp_cols],
            "Promedio (0–5)": [df_f[b].mean() if b in df_f.columns else np.nan for _, b in comp_cols],
        }
    ).dropna().sort_values("Promedio (0–5)", ascending=True)

    fig_comp = px.bar(
        comp_df,
        x="Promedio (0–5)",
        y="Componente",
        orientation="h",
        text="Promedio (0–5)",
        color="Promedio (0–5)",
        color_continuous_scale=["#ff4d4f", "#faad14", "#52c41a", "#722ed1"],
    )
    fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_comp.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.info(
        "Tip: este gráfico te dice *en qué* mejorar. "
        "Los componentes más bajos son los mejores candidatos para un plan de acción (gobierno de datos, BI, estandarización de reportes, etc.)."
    )

# -------------------------
# TAB 3: Detalle
# -------------------------
with tab3:
    st.subheader("📋 Detalle (por defecto: toda la organización)")
    cols_show = [
        "Área",
        "Nombre1",
        "Madurez_0_100",
        "Nivel_madurez",
        "Nivel promedio de Excel",
        "¿Se usa Power BI en el área?",
        "¿Qué tan seguido se crean reportes en tu área?",
        "¿Qué tan seguido encuentran problemas con relación a datos?",
        "En reuniones de seguimiento, ¿qué tanto se decide con datos vs “percepción”?",
        "¿Qué tan familiarizo estás con herramientas de IA? 1-5",
        "¿Cuál descripción se parece más a tu área hoy?",
    ]
    cols_show = [c for c in cols_show if c in df_f.columns]

    if show_people:
        sort_cols = ["Área", "Madurez_0_100"] if "Área" in df_f.columns else ["Madurez_0_100"]
        df_out = df_f[cols_show].sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else [False])
        st.dataframe(df_out, use_container_width=True, height=520)
    else:
        st.dataframe(by_area, use_container_width=True, height=520)

    st.caption(
        "🔎 Los filtros del sidebar son opcionales: el tablero siempre arranca mostrando toda la información. "
        "El índice Madurez_0_100 se calcula mapeando respuestas a escala 1–5 y promediando componentes."
    )



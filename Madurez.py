import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Dashboard Madurez Digital", layout="wide")
st.title("📊 Dashboard — Madurez Digital, Tecnología y Analítica")

# GitHub-friendly: el Excel vive en la raíz del repo
EXCEL_PATH = "Madurez Digital, Tecnología y Analítica.xlsx"
SHEET_NAME = "DATA"


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
    # Forzamos openpyxl para leer .xlsx
    return pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")


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
    col_want_ai = "¿Te gustaría implementar proyectos de inteligencia artificial para automatizar tareas de tu área?"

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

    # Scores
    d["score_dependencia"] = d[col_dep].map(lambda x: map_dep.get(norm_text(x), np.nan))
    d["score_excel"] = d[col_excel].map(lambda x: map_excel.get(norm_text(x), np.nan))
    d["score_powerbi"] = d[col_pbi].map(lambda x: map_pbi.get(norm_text(x), np.nan))
    d["score_dashboards"] = pd.to_numeric(d[col_dash], errors="coerce")
    d["score_freq_reportes"] = d[col_rep].map(lambda x: map_rep.get(norm_text(x), np.nan))
    d["score_dataset_base"] = d[col_dataset].map(yes_no)
    d["score_calidad_datos"] = d[col_prob].map(lambda x: map_prob.get(norm_text(x), np.nan))
    d["score_decisiones_con_datos"] = d[col_dec].map(lambda x: map_dec.get(norm_text(x), np.nan))
    d["score_ideas_mejora"] = d[col_ideas].map(lambda x: map_ideas.get(norm_text(x), np.nan))
    d["score_conoce_herr_digitales"] = d[col_tools].map(yes_no)
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

    return d


# =====================================================
# CARGA
# =====================================================
try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(
        "❌ No encontré el Excel en la raíz del repo.\n\n"
        f"Asegúrate de que exista este archivo:\n\n`{EXCEL_PATH}`\n\n"
        "y que el sheet se llame `DATA`."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error leyendo el Excel con openpyxl:\n\n{e}")
    st.stop()

df = add_scores(df_raw)

# =====================================================
# FILTROS
# =====================================================
with st.sidebar:
    st.header("🔎 Filtros")
    areas = sorted(df["Área"].dropna().unique().tolist()) if "Área" in df.columns else []
    if areas:
        selected_areas = st.multiselect("Área", options=areas, default=areas)
    else:
        selected_areas = []
        st.warning("No se encontró la columna 'Área'. Se mostrarán todos los datos.")
    show_people = st.checkbox("Mostrar tabla por persona", value=True)

df_f = df[df["Área"].isin(selected_areas)].copy() if selected_areas else df.copy()

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

k1, k2, k3, k4 = st.columns(4)
k1.metric("Madurez promedio (0–100)", f"{overall:.1f}" if not np.isnan(overall) else "-")
k2.metric("Respuestas", f"{n_resp}")
k3.metric("Área más madura", f"{best_area}")
k4.metric("Power BI en uso (aprox.)", f"{pbi_yes:.0f}%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Dataset base existente (aprox.)", f"{dataset_yes:.0f}%")
k6.metric("Área menos madura", f"{worst_area}")
k7.metric("Mediana madurez", f"{df_f['Madurez_0_100'].median():.1f}" if len(df_f) else "-")
k8.metric("Máx / Mín", f"{df_f['Madurez_0_100'].max():.1f} / {df_f['Madurez_0_100'].min():.1f}" if len(df_f) else "-")

st.divider()

# =====================================================
# GRÁFICAS
# =====================================================
c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("🏢 Madurez promedio por área")
    if len(by_area):
        fig_area = px.bar(
            by_area,
            x="Madurez_0_100",
            y="Área",
            orientation="h",
            text="Madurez_0_100",
        )
        fig_area.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_area.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No hay datos suficientes para graficar por área.")

with c2:
    st.subheader("📈 Distribución de madurez (personas)")
    fig_hist = px.histogram(df_f, x="Madurez_0_100", nbins=12)
    fig_hist.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🧩 Excel / Power BI / Calidad de datos")
    col_excel = "Nivel promedio de Excel"
    if col_excel in df_f.columns:
        fig_excel = px.bar(df_f[col_excel].value_counts().reset_index(), x=col_excel, y="count")
        fig_excel.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_excel, use_container_width=True)

    if pbi_col in df_f.columns:
        fig_pbi = px.bar(df_f[pbi_col].value_counts().reset_index(), x=pbi_col, y="count")
        fig_pbi.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pbi, use_container_width=True)

    col_prob = "¿Qué tan seguido encuentran problemas con relación a datos?"
    if col_prob in df_f.columns:
        fig_prob = px.bar(df_f[col_prob].value_counts().reset_index(), x=col_prob, y="count")
        fig_prob.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prob, use_container_width=True)

st.divider()

# =====================================================
# COMPONENTES (PROMEDIOS)
# =====================================================
st.subheader("🧠 Promedios de componentes (0–5)")
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
).dropna().sort_values("Promedio (0–5)", ascending=False)

if len(comp_df):
    fig_comp = px.bar(comp_df, x="Promedio (0–5)", y="Componente", orientation="h", text="Promedio (0–5)")
    fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_comp.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No se pudieron calcular promedios de componentes (revisa columnas del Excel).")

st.divider()

# =====================================================
# TABLA
# =====================================================
st.subheader("📋 Respuestas (filtradas)")
cols_show = [
    "Área",
    "Nombre1",
    "Madurez_0_100",
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
    st.dataframe(
        df_f[cols_show].sort_values(
            ["Área", "Madurez_0_100"] if "Área" in df_f.columns else ["Madurez_0_100"],
            ascending=[True, False] if "Área" in df_f.columns else [False],
        ),
        use_container_width=True,
    )
else:
    st.dataframe(by_area, use_container_width=True)

st.caption("Nota: El índice Madurez_0_100 se calcula mapeando respuestas a una escala 1–5 y promediando componentes.")


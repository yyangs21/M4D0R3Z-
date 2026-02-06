import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Dashboard Madurez Digital", layout="wide")

EXCEL_PATH = "Madurez Digital.xlsx"  # en la raíz del repo
SHEET_NAME = "DATA"

PALETTE = px.colors.qualitative.Set2


# =====================================================
# HELPERS
# =====================================================
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def is_na_area(x) -> bool:
    t = norm_text(x)
    return t in {"n/a", "na", "n.a", "n.a.", "none", "sin área", "sin area", "", "-", "null"}


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
    map_prob = {"muy frecuente": 2, "frecuente": 3, "a veces": 4, "casi nunca": 5, "nunca": 5}
    map_dec = {
        "casi todo percepción": 1,
        "casi todo percepcion": 1,
        "mitad y mitad": 3,
        "mayormente con datos": 4,
        "siempre con datos": 5,
    }
    map_ideas = {"nunca": 1, "rara vez": 2, "a veces": 3, "frecuente": 4, "siempre": 5}

    # Scores (0–5)
    d["score_dependencia"] = d[col_dep].map(lambda x: map_dep.get(norm_text(x), np.nan))
    d["score_excel"] = d[col_excel].map(lambda x: map_excel.get(norm_text(x), np.nan))
    d["score_powerbi"] = d[col_pbi].map(lambda x: map_pbi.get(norm_text(x), np.nan))
    d["score_dashboards"] = pd.to_numeric(d[col_dash], errors="coerce")
    d["score_freq_reportes"] = d[col_rep].map(lambda x: map_rep.get(norm_text(x), np.nan))

    # Binarios -> escala suave (para que aporte sin dominar)
    d["score_dataset_base"] = d[col_dataset].map(yes_no).map(lambda v: 5 if v == 1 else (2 if v == 0 else np.nan))
    d["score_conoce_herr_digitales"] = d[col_tools].map(yes_no).map(lambda v: 4 if v == 1 else (2 if v == 0 else np.nan))

    d["score_calidad_datos"] = d[col_prob].map(lambda x: map_prob.get(norm_text(x), np.nan))
    d["score_decisiones_con_datos"] = d[col_dec].map(lambda x: map_dec.get(norm_text(x), np.nan))
    d["score_ideas_mejora"] = d[col_ideas].map(lambda x: map_ideas.get(norm_text(x), np.nan))
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
        f"❌ No encontré el Excel en la raíz del repo: `{EXCEL_PATH}`\n\n"
        f"Verifica también que el sheet se llame `{SHEET_NAME}`."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error leyendo el Excel con openpyxl:\n\n{e}")
    st.stop()

df = add_scores(df_raw)

# Normalización básica (evita espacios raros)
for col in ["Área", "Departamento", "Nombre1"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# =====================================================
# SIDEBAR: filtros (por defecto TODO)
# =====================================================
st.sidebar.title("🔎 Filtros (opcional)")

# ÁREAS disponibles (excluye N/A para opciones y filtro)
areas_valid = []
if "Área" in df.columns:
    areas_valid = sorted([a for a in df["Área"].dropna().unique().tolist() if not is_na_area(a)])

depts = []
if "Departamento" in df.columns:
    depts = sorted(df["Departamento"].dropna().unique().tolist())

selected_depts = st.sidebar.multiselect("Departamento", options=depts, default=depts) if depts else []
selected_areas = st.sidebar.multiselect("Área", options=areas_valid, default=areas_valid) if areas_valid else []

show_people = st.sidebar.checkbox("Ver tabla por persona", value=True)

# Aplicación de filtros (Departamento y Área)
df_f = df.copy()
if selected_depts and "Departamento" in df_f.columns:
    df_f = df_f[df_f["Departamento"].isin(selected_depts)].copy()
# Nota: el filtro de área SOLO deja pasar áreas válidas (ya no hay N/A por definición)
if selected_areas and "Área" in df_f.columns:
    df_f = df_f[df_f["Área"].isin(selected_areas)].copy()

# Dataset SOLO para gráficos por ÁREA (excluye registros con Área = N/A)
df_area = df_f.copy()
if "Área" in df_area.columns:
    df_area = df_area[~df_area["Área"].apply(is_na_area)].copy()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
<div style="padding: 14px 16px; border-radius: 14px; background: rgba(0,0,0,0.04);">
  <h2 style="margin:0;">📌 Resumen Ejecutivo</h2>
  <div style="margin-top:6px; color: rgba(0,0,0,0.65);">
    El tablero inicia mostrando <b>toda la información</b>.  
    Los filtros (Departamento / Área) son opcionales para enfocarte en una sección específica.  
    <br/><b>Importante:</b> los registros con <b>Área = N/A</b> no se consideran en las gráficas y rankings por Área.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# =====================================================
# KPIs (global con filtros aplicados)
# =====================================================
overall = float(df_f["Madurez_0_100"].mean()) if len(df_f) else np.nan
n_resp = int(len(df_f))

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

# Rankings por Área (SIN N/A) y por Departamento
by_area = (
    df_area.groupby("Área", as_index=False)["Madurez_0_100"].mean().sort_values("Madurez_0_100", ascending=False)
    if "Área" in df_area.columns and len(df_area)
    else pd.DataFrame(columns=["Área", "Madurez_0_100"])
)
by_dept = (
    df_f.groupby("Departamento", as_index=False)["Madurez_0_100"].mean().sort_values("Madurez_0_100", ascending=False)
    if "Departamento" in df_f.columns and len(df_f)
    else pd.DataFrame(columns=["Departamento", "Madurez_0_100"])
)

best_area = by_area.iloc[0]["Área"] if len(by_area) else "-"
worst_area = by_area.iloc[-1]["Área"] if len(by_area) else "-"
best_dept = by_dept.iloc[0]["Departamento"] if len(by_dept) else "-"
worst_dept = by_dept.iloc[-1]["Departamento"] if len(by_dept) else "-"

c1, c2, c3, c4, c5 = st.columns([1.25, 1, 1, 1, 1])

with c1:
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
    st.plotly_chart(fig_gauge, use_container_width=True, key="kpi_gauge")

with c2:
    st.metric("Respuestas", f"{n_resp}")
    st.metric("Mediana", f"{df_f['Madurez_0_100'].median():.1f}" if len(df_f) else "-")

with c3:
    st.metric("Área más madura", best_area)
    st.metric("Área menos madura", worst_area)

with c4:
    st.metric("Depto más maduro", best_dept)
    st.metric("Depto menos maduro", worst_dept)

with c5:
    st.metric("Power BI en uso", f"{pbi_yes:.0f}%")
    st.metric("Dataset base", f"{dataset_yes:.0f}%")

st.divider()

# =====================================================
# TABS
# =====================================================
tab_area, tab_dept, tab_comp, tab_det = st.tabs(
    ["📊 Por área", "🏢 Por departamento", "🧩 Componentes", "📋 Detalle"]
)

# -------------------------
# TAB: ÁREA (SIN N/A)
# -------------------------
with tab_area:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("🏢 Ranking de madurez por Área (promedio) — excluye Área=N/A")
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
            st.plotly_chart(fig_area, use_container_width=True, key="chart_area_ranking")
        else:
            st.info("No hay datos suficientes para graficar por Área (sin N/A).")

    with right:
        st.subheader("📈 Distribución (personas)")
        fig_hist = px.histogram(df_area if len(df_area) else df_f, x="Madurez_0_100", nbins=12, color_discrete_sequence=[PALETTE[0]])
        fig_hist.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True, key="chart_area_hist")

        st.subheader("🚦 Semáforo (conteo)")
        order = ["🔴 Baja", "🟡 Media", "🟢 Buena", "🟣 Alta"]
        sem = (df_area if len(df_area) else df_f)["Nivel_madurez"].value_counts().reindex(order).fillna(0).reset_index()
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
        st.plotly_chart(fig_sem, use_container_width=True, key="chart_area_semaforo")

# -------------------------
# TAB: DEPARTAMENTO
# -------------------------
with tab_dept:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("🏢 Ranking de madurez por Departamento (promedio)")
        if len(by_dept):
            fig_dept = px.bar(
                by_dept,
                x="Madurez_0_100",
                y="Departamento",
                orientation="h",
                text="Madurez_0_100",
                color="Madurez_0_100",
                color_continuous_scale=["#ff4d4f", "#faad14", "#52c41a", "#722ed1"],
            )
            fig_dept.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_dept.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_dept, use_container_width=True, key="chart_dept_ranking")
        else:
            st.info("No hay datos suficientes para graficar por Departamento (revisa la columna 'Departamento').")

    with right:
        st.subheader("📈 Distribución (personas)")
        fig_hist2 = px.histogram(df_f, x="Madurez_0_100", nbins=12, color_discrete_sequence=[PALETTE[2]])
        fig_hist2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hist2, use_container_width=True, key="chart_dept_hist")

        st.subheader("🚦 Semáforo (conteo)")
        order = ["🔴 Baja", "🟡 Media", "🟢 Buena", "🟣 Alta"]
        sem2 = df_f["Nivel_madurez"].value_counts().reindex(order).fillna(0).reset_index()
        sem2.columns = ["Nivel", "Personas"]
        fig_sem2 = px.bar(
            sem2,
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
        fig_sem2.update_traces(textposition="outside")
        fig_sem2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sem2, use_container_width=True, key="chart_dept_semaforo")

# -------------------------
# TAB: COMPONENTES
# -------------------------
with tab_comp:
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

    if len(comp_df):
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
        st.plotly_chart(fig_comp, use_container_width=True, key="chart_componentes")
    else:
        st.info("No se pudieron calcular componentes (revisa columnas del Excel).")

# -------------------------
# TAB: DETALLE
# -------------------------
with tab_det:
    st.subheader("📋 Detalle (con filtros aplicados)")

    cols_show = [
        "Departamento",
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
        sort_cols = []
        if "Departamento" in df_f.columns:
            sort_cols.append("Departamento")
        if "Área" in df_f.columns:
            sort_cols.append("Área")
        sort_cols.append("Madurez_0_100")

        df_out = df_f[cols_show].sort_values(
            sort_cols,
            ascending=[True] * (len(sort_cols) - 1) + [False]
        )
        st.dataframe(df_out, use_container_width=True, height=560, key="tabla_detalle")
    else:
        st.write("**Resumen por Departamento**")
        st.dataframe(by_dept, use_container_width=True, height=260, key="tabla_resumen_dept")
        st.write("**Resumen por Área (excluye N/A)**")
        st.dataframe(by_area, use_container_width=True, height=260, key="tabla_resumen_area")

    st.caption(
        "Los filtros son opcionales."
            )




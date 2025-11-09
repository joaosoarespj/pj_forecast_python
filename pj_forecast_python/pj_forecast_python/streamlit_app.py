import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import glob, os

# ===============================
# CONFIGURAÇÕES INICIAIS
# ===============================
st.set_page_config(page_title="PJ Forecast Dashboard", layout="wide")
st.title("Forecast (Markov + Tickets) — Painel Analítico")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECASTS_DIR = os.path.join(ROOT_DIR, "artifacts", "forecasts")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(FORECASTS_DIR, exist_ok=True)

# ===============================
# CARREGAMENTO DE DADOS
# ===============================
forecasts = sorted(glob.glob(os.path.join(FORECASTS_DIR, "summary_*.csv")))
if not forecasts:
    st.warning("Nenhum forecast encontrado. Rode python -m pj_forecast_python.scripts.run_forecast primeiro.")
    st.stop()

selected = st.selectbox(
    "Selecione a versão do forecast:",
    forecasts,
    index=len(forecasts) - 1,
    format_func=os.path.basename
)

df_forecast = pd.read_csv(selected)
df_forecast.columns = [c.strip().lower() for c in df_forecast.columns]
for col in ("p10", "p60", "p90"):
    if col in df_forecast.columns:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce")

# versão anterior (para comparação)
previous = None
if len(forecasts) > 1:
    prev_path = forecasts[-2]
    previous = pd.read_csv(prev_path)
    previous.columns = [c.strip().lower() for c in previous.columns]
    for col in ("p10", "p60", "p90"):
        if col in previous.columns:
            previous[col] = pd.to_numeric(previous[col], errors="coerce")

# Base histórica (realizado)
real_path = os.path.join(DATA_DIR, "base_historica.csv")
if os.path.exists(real_path):
    df_real = pd.read_csv(real_path)
    df_real.columns = [c.lower() for c in df_real.columns]
    df_real["data"] = pd.to_datetime(df_real["data"], errors="coerce")
    df_real["ano"] = df_real["data"].dt.year
    df_real["mes"] = df_real["data"].dt.month
    for c in ("vendas", "ticket"):
        if c in df_real.columns:
            df_real[c] = pd.to_numeric(df_real[c], errors="coerce")
else:
    st.warning("⚠️ Nenhum arquivo encontrado em data/base_historica.csv")
    df_real = pd.DataFrame(columns=["data", "ticket", "vendas", "ano", "mes", "estado"])

# ===============================
# FILTROS
# ===============================
st.sidebar.header("Filtros")

anos_disp = sorted(df_real["ano"].dropna().unique()) if "ano" in df_real.columns else []
ano_filtro = st.sidebar.selectbox(
    "Ano de comparação (realizado)",
    options=anos_disp if anos_disp else [2025],
    index=len(anos_disp) - 1 if anos_disp else 0
)

estados = ["Todos"]
if "estado" in df_real.columns and not df_real.empty:
    estados += sorted(df_real["estado"].dropna().unique())
estado_filtro = st.sidebar.selectbox("Estado (para o realizado)", estados, index=0)

# ===============================
# SEPARAÇÃO FORECAST
# ===============================
metric_col = "metric" if "metric" in df_forecast.columns else None
if metric_col:
    monthly_mask = ~df_forecast[metric_col].str.contains("annual", case=False, na=False)
    df_monthly = df_forecast[monthly_mask].copy().head(12)
    df_annual = df_forecast[~monthly_mask].copy()
else:
    df_monthly = df_forecast.head(12).copy()
    df_annual = pd.DataFrame()

df_monthly = df_monthly.reset_index(drop=True)
df_monthly["mes"] = range(1, len(df_monthly) + 1)

# anuais reais
p10_annual = float(df_annual["p10"].iloc[0]) if "p10" in df_annual.columns and not df_annual.empty else np.nan
p60_annual = float(df_annual["p60"].iloc[0]) if "p60" in df_annual.columns and not df_annual.empty else np.nan
p90_annual = float(df_annual["p90"].iloc[0]) if "p90" in df_annual.columns and not df_annual.empty else np.nan

# ===============================
# REALIZADO = vendas * ticket
# ===============================
df_real_filt = df_real.copy()
df_real_filt = df_real_filt[df_real_filt["ano"] == ano_filtro]
if estado_filtro != "Todos" and "estado" in df_real_filt.columns:
    df_real_filt = df_real_filt[df_real_filt["estado"] == estado_filtro]

if {"vendas", "ticket", "mes"}.issubset(df_real_filt.columns):
    df_real_filt["valor"] = df_real_filt["vendas"] * df_real_filt["ticket"]
    real_mensal = (
        df_real_filt.groupby("mes", as_index=False)["valor"].sum().sort_values("mes")
    )
    real_mensal["acumulado"] = real_mensal["valor"].cumsum()
else:
    real_mensal = pd.DataFrame(columns=["mes", "valor", "acumulado"])

# ===============================
# ABAS
# ===============================
aba = st.tabs([
    "Evolução acumulada",
    "Sazonalidade mensal",
    "Comparação entre versões",
    "Distribuição / Simulação",
    "Resumo anual"
])

# ===========================================================
# ABA 1 — EVOLUÇÃO ACUMULADA (ajustada)
# ===========================================================
with aba[0]:
    st.subheader("Evolução do Faturamento — Forecast 2026 vs Realizado 2025")

    if all(col in df_monthly.columns for col in ["p10", "p60", "p90"]):
        df_sazonal = df_monthly.copy()
        df_sazonal["mes"] = range(1, len(df_sazonal) + 1)

        # proporção mensal (mantém forma)
        soma_p10 = df_sazonal["p10"].sum()
        soma_p60 = df_sazonal["p60"].sum()
        soma_p90 = df_sazonal["p90"].sum()

        # redistribui para bater com valores anuais reais
        df_sazonal["p10_adj"] = (df_sazonal["p10"] / soma_p10) * p10_annual if soma_p10 > 0 else 0
        df_sazonal["p60_adj"] = (df_sazonal["p60"] / soma_p60) * p60_annual if soma_p60 > 0 else 0
        df_sazonal["p90_adj"] = (df_sazonal["p90"] / soma_p90) * p90_annual if soma_p90 > 0 else 0

        # acumulado ajustado
        df_sazonal["P10_acumulado"] = df_sazonal["p10_adj"].cumsum()
        df_sazonal["P60_acumulado"] = df_sazonal["p60_adj"].cumsum()
        df_sazonal["P90_acumulado"] = df_sazonal["p90_adj"].cumsum()

        if not real_mensal.empty:
            df_sazonal = df_sazonal.merge(real_mensal[["mes", "acumulado"]], on="mes", how="left")
            df_sazonal.rename(columns={"acumulado": "Realizado_acum"}, inplace=True)

        # gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sazonal["mes"], y=df_sazonal["P10_acumulado"], name="P10", line=dict(color="lightblue")))
        fig.add_trace(go.Scatter(x=df_sazonal["mes"], y=df_sazonal["P60_acumulado"], name="P60 (central)", line=dict(color="royalblue", width=3)))
        fig.add_trace(go.Scatter(x=df_sazonal["mes"], y=df_sazonal["P90_acumulado"], name="P90", line=dict(color="lightblue")))

        if "Realizado_acum" in df_sazonal.columns:
            fig.add_trace(go.Scatter(
                x=df_sazonal["mes"], y=df_sazonal["Realizado_acum"],
                name=f"Realizado {ano_filtro}",
                line=dict(color="orange", dash="dash")
            ))

        # marcadores finais
        fig.add_trace(go.Scatter(x=[12], y=[p10_annual], mode="markers+text",
                                 text=["P10 anual"], textposition="bottom right",
                                 marker_symbol="diamond", marker_size=10, marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=[12], y=[p60_annual], mode="markers+text",
                                 text=["P60 anual"], textposition="bottom right",
                                 marker_symbol="diamond", marker_size=10, marker_color="royalblue"))
        fig.add_trace(go.Scatter(x=[12], y=[p90_annual], mode="markers+text",
                                 text=["P90 anual"], textposition="bottom right",
                                 marker_symbol="diamond", marker_size=10, marker_color="lightblue"))

        fig.update_layout(
            title="Faturamento acumulado — calibrado para bater com os valores anuais do forecast",
            xaxis_title="Mês",
            yaxis_title="Faturamento acumulado (R$)",
            template="plotly_dark",
            height=560
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("As curvas foram recalculadas proporcionalmente para que o mês 12 coincida exatamente com os valores anuais reais (P10 / P60 / P90).")

# ===========================================================
# ABA 2 — SAZONALIDADE MENSAL
# ===========================================================
with aba[1]:
    st.subheader("Sazonalidade mensal — P10 / P60 / P90 vs Realizado")

    fig2 = go.Figure()
    if all(col in df_monthly.columns for col in ["p10", "p60", "p90"]):
        fig2.add_trace(go.Scatter(x=df_monthly["mes"], y=df_monthly["p10"], mode="lines", name="P10", line=dict(color="lightblue")))
        fig2.add_trace(go.Scatter(x=df_monthly["mes"], y=df_monthly["p60"], mode="lines+markers", name="P60 (central)", line=dict(color="royalblue", width=3)))
        fig2.add_trace(go.Scatter(x=df_monthly["mes"], y=df_monthly["p90"], mode="lines", name="P90", line=dict(color="lightblue")))

    if not real_mensal.empty:
        fig2.add_trace(go.Scatter(
            x=real_mensal["mes"], y=real_mensal["valor"],
            mode="lines+markers", name=f"Realizado {ano_filtro} (vendas×ticket)",
            line=dict(color="orange", dash="dash")
        ))

    fig2.update_layout(
        title="Sazonalidade mensal (valores de faturamento)",
        xaxis_title="Mês",
        yaxis_title="Faturamento mensal (R$)",
        template="plotly_dark",
        height=520
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================
# COMPARAÇÃO ENTRE VERSÕES
# ===========================================================
with aba[2]:
    st.subheader("Comparativo entre versões (Δ P60)")

    if previous is not None and "p60" in df_monthly.columns and "p60" in previous.columns:
        prev_monthly = previous.copy().head(len(df_monthly))
        df_comp = pd.DataFrame({
            "mês": df_monthly["mes"],
            "P60_atual": df_monthly["p60"],
            "P60_anterior": prev_monthly["p60"],
        })
        df_comp["Δ %"] = 100 * (df_comp["P60_atual"] - df_comp["P60_anterior"]) / df_comp["P60_anterior"].replace({0: np.nan})

        st.dataframe(df_comp.style.format({"Δ %": "{:.2f}%"}), use_container_width=True)
        fig3 = px.bar(df_comp, x="mês", y="Δ %", color="Δ %", color_continuous_scale="Blues",
                      title="Variação percentual do P60 entre versões")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Não há versão anterior para comparar.")

# ===========================================================
# DISTRIBUIÇÃO / SIMULAÇÃO
# ===========================================================
with aba[3]:
    st.subheader("Distribuição (Densidade de Simulações)")

    if all(col in df_monthly.columns for col in ["p10", "p60", "p90"]) and len(df_monthly) > 0:
        dist = []
        for i in range(len(df_monthly)):
            mu = df_monthly.loc[i, "p60"]
            span = df_monthly.loc[i, "p90"] - df_monthly.loc[i, "p10"]
            if pd.notna(mu) and pd.notna(span) and span > 0:
                sigma = span / 2.6
                sim = np.random.normal(mu, sigma, 5000)
                dist.extend(sim)
        if len(dist) > 0:
            fig4 = px.histogram(dist, nbins=40, title="Distribuição simulada de resultados (P10–P90)",
                                template="plotly_dark", color_discrete_sequence=["royalblue"])
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Não foi possível simular a distribuição (valores insuficientes).")

# ===========================================================
# RESUMO ANUAL
# ===========================================================
with aba[4]:
    st.subheader("Resumo Anual — P10 / P60 / P90")
    col1, col2, col3 = st.columns(3)
    col1.metric("P10 anual", f"{p10_annual:,.0f}".replace(",", ".") if not np.isnan(p10_annual) else "—")
    col2.metric("P60 anual", f"{p60_annual:,.0f}".replace(",", ".") if not np.isnan(p60_annual) else "—")
    col3.metric("P90 anual", f"{p90_annual:,.0f}".replace(",", ".") if not np.isnan(p90_annual) else "—")

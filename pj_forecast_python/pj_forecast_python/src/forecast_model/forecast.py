import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .utils import (
    add_time_cols,
    classificar_estado_anual,
    percentil_ponderado_por_estado,
)

# --------------------------------------
# 1) Projeção de vendas (sazonalidade + tendência)
# --------------------------------------

def _fatores_sazonais_por_mes(df: pd.DataFrame) -> Dict[int, float]:
    """
    Calcula fatores sazonais mensais (multiplicativos) usando anos mais recentes com maior peso.
    Estratégia:
      1) média de vendas por ano -> tendência anual
      2) fator_mensal = vendas_mes / média_anual daquele ano
      3) média ponderada dos fatores por mês (anos recentes pesam mais)
    """
    df = df.copy()
    if "ano" not in df.columns or "mes" not in df.columns:
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month

    # usa apenas anos com alguma atividade
    anos = sorted(a for a in df["ano"].unique())
    if not anos:
        return {m: 1.0 for m in range(1, 13)}

    # pesos por ano (recência maior peso)
    base = np.linspace(1.0, 1.0 + 0.2 * (len(anos) - 1), num=len(anos))
    w = {ano: float(base[i] / base.sum()) for i, ano in enumerate(anos)}

    fatores = {m: [] for m in range(1, 13)}
    pesos   = {m: [] for m in range(1, 13)}

    for ano in anos:
        sub = df[df["ano"] == ano]
        if sub.empty: 
            continue
        media_anual = sub["vendas"].mean() if sub["vendas"].notna().any() else 0.0
        if media_anual <= 0:
            # se não tem vendas no ano, pula para evitar dividir por zero
            continue
        for m in range(1, 13):
            vm = float(sub.loc[sub["mes"] == m, "vendas"].mean()) if (sub["mes"] == m).any() else np.nan
            if not np.isnan(vm):
                fatores[m].append(vm / media_anual)
                pesos[m].append(w[ano])

    fatores_mensais = {}
    for m in range(1, 13):
        if not fatores[m]:
            fatores_mensais[m] = 1.0
        else:
            f = np.average(np.array(fatores[m]), weights=np.array(pesos[m]))
            fatores_mensais[m] = float(f)

    # normaliza para somarem ~12 (média ~1.0)
    norm = 12.0 / sum(fatores_mensais.values())
    for m in fatores_mensais:
        fatores_mensais[m] *= norm

    return fatores_mensais


def _tendencia_anual(df: pd.DataFrame) -> float:
    """
    Estima taxa de crescimento anual (g) pela regressão linear simples das médias anuais.
    Retorna fator de crescimento relativo por ano (ex.: 0.10 = +10% a/a).
    Caso série curta, retorna 0.0.
    """
    ano_means = df.groupby("ano")["vendas"].mean().reset_index()
    if len(ano_means) < 2:
        return 0.0

    # regressão simples y = a + b*t
    ano_means = ano_means.sort_values("ano")
    t = np.arange(len(ano_means))
    y = ano_means["vendas"].values.astype(float)

    # b = cov(t,y) / var(t)
    b = np.cov(t, y, ddof=0)[0,1] / (np.var(t) + 1e-9)
    # crescimento relativo aproximado por período
    if ano_means["vendas"].mean() <= 0:
        return 0.0
    g = b / ano_means["vendas"].mean()
    return float(g)


def projetar_vendas_para_proximo_ano(df_hist: pd.DataFrame) -> pd.Series:
    """
    Cria vetor de vendas projetadas para os 12 meses seguintes ao último ano disponível.
    Usa fatores sazonais mensais e tendência anual.
    Resultado: pd.Series index 1..12 com as vendas projetadas (floats).
    """
    df = add_time_cols(df_hist)
    fatores = _fatores_sazonais_por_mes(df)
    g = _tendencia_anual(df)

    ultimo_ano = int(df["ano"].max())
    # baseline = média anual do último ano não-nulo; se zero, usa média dos últimos 24 meses não-zero
    sub_last = df[df["ano"] == ultimo_ano]
    media_last = sub_last["vendas"].replace(0, np.nan).mean()
    if np.isnan(media_last) or media_last <= 0:
        media_last = df["vendas"].replace(0, np.nan).tail(24).mean()
    if np.isnan(media_last) or media_last <= 0:
        media_last = max(1.0, df["vendas"].mean())

    # aplica crescimento para o próximo ano inteiro
    fator_crescimento = 1.0 + max(min(g, 0.50), -0.50)  # limita em ±50%/ano
    vendas_proj = {}
    for m in range(1, 13):
        vendas_proj[m] = float(media_last * fatores[m] * fator_crescimento)

    # garante mínimo inteiro e >=0
    return pd.Series(vendas_proj).clip(lower=0.0)


# --------------------------------------
# 2) Markov – mensal + viés trimestral
# --------------------------------------

def _matriz_transicao_estados(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Estima matriz de transição entre estados (global, mês->mês) e distribuição inicial.
    estados: {'baixo','medio','alto'}
    """
    estados = ["baixo", "medio", "alto"]
    counts = {e: {f: 1.0 for f in estados} for e in estados}  # suavização Laplace
    pi = {e: 1.0 for e in estados}

    df = df.sort_values("data").reset_index(drop=True)
    for ano, sub in df.groupby("ano"):
        sub = sub.sort_values("data")
        est = sub["estado"].astype(str).values
        for i in range(len(est) - 1):
            a, b = est[i], est[i+1]
            if a in counts and b in counts[a]:
                counts[a][b] += 1.0
            if i == 0 and a in pi:
                pi[a] += 1.0

    # normaliza
    P = {}
    for a in estados:
        total = sum(counts[a].values())
        P[a] = {b: counts[a][b] / total for b in estados}

    total_pi = sum(pi.values())
    pi = {e: pi[e] / total_pi for e in estados}
    return P, pi


def _bias_trimestral(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Calcula distribuição média de estados por TRIMESTRE (1..4),
    usada como viés multiplicativo (reponderação leve) nas probabilidades mensais.
    """
    df = df.copy()
    if "trimestre" not in df.columns:
        df["trimestre"] = df["data"].dt.quarter

    estados = ["baixo", "medio", "alto"]
    bias = {t: {e: 1/3 for e in estados} for t in [1,2,3,4]}

    for t, sub in df.groupby("trimestre"):
        base = sub["estado"].value_counts(normalize=True)
        for e in estados:
            if e in base.index:
                bias[t][e] = float(base[e])

        # suavização leve
        s = sum(bias[t].values())
        for e in estados:
            bias[t][e] = (bias[t][e] + 0.05) / (s + 0.15)

    return bias


def _amostrar_estado(atual: str, P: Dict[str, Dict[str, float]], viés: Dict[str, float], rng: np.random.Generator) -> str:
    """
    Amostra próximo estado a partir de P[atual] reponderado pelo viés do trimestre.
    """
    estados = ["baixo", "medio", "alto"]
    probs = np.array([P[atual][e] * viés.get(e, 1.0) for e in estados], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(estados, p=probs)


# --------------------------------------
# 3) Simulação principal
# --------------------------------------

def generate_forecast(
    df_hist: pd.DataFrame,
    n_sims: int = 5000,
    horizon_months: int = 12,
    percentile_ticket: float = 0.60,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      - df_sims_long: colunas [sim, mes, estado, ticket, vendas, faturamento]
      - df_summary:   12 linhas mensais com p10/p60/p90 + 1 linha 'annual'
    """
    rng = np.random.default_rng(seed)

    # 1) preparar dados
    df = df_hist.copy()
    df = add_time_cols(df)
    if "estado" not in df.columns:
        df = classificar_estado_anual(df)

    # 2) tickets representativos por estado (≥2024 com pesos; 2026 terá peso maior assim que existir)
    reps = percentil_ponderado_por_estado(df, ano_min=2024, percentile=percentile_ticket)

    # 3) estatística de dispersão por estado (dp ponderado simples)
    disp = {}
    for e, sub in df.groupby("estado"):
        vals = sub["ticket"].replace(0, np.nan).dropna().values
        disp[e] = float(np.nanstd(vals)) if len(vals) else 0.0
        # bound mínimo para evitar zero total
        if disp[e] == 0.0:
            disp[e] = max(0.05 * reps.get(e, 0.0), 1.0)

    # 4) Markov mensal e viés trimestral
    P, pi = _matriz_transicao_estados(df)
    bias_T = _bias_trimestral(df)

    # 5) Projeção de vendas para o PRÓXIMO ano (12 meses)
    vendas_proj = projetar_vendas_para_proximo_ano(df)

    # 6) Simulação Monte Carlo
    sims = []
    estados = ["baixo", "medio", "alto"]
    start_estado = rng.choice(estados, p=np.array([pi[e] for e in estados], dtype=float))

    for s in range(n_sims):
        estado = start_estado
        for m in range(1, horizon_months + 1):
            trimestre = ((m - 1) // 3) + 1  # 1..4
            viés = bias_T.get(trimestre, {e: 1/3 for e in estados})

            # sorteia ticket a partir do representante + ruído positivo (trunc normal)
            mu = reps.get(estado, 0.0)
            sd = disp.get(estado, 1.0)
            # amostra normal e trunca no zero
            t = rng.normal(loc=mu, scale=sd)
            if t < 0: t = abs(t) * 0.5
            vendas_m = float(vendas_proj.get(m, 0.0))
            faturamento = float(t * vendas_m)

            sims.append((s, m, estado, float(t), vendas_m, faturamento))

            # próximo estado (Markov com viés trimestral)
            estado = _amostrar_estado(estado, P, viés, rng)

    df_sims = pd.DataFrame(sims, columns=["sim", "mes", "estado", "ticket", "vendas", "faturamento"])

    # 7) Resumos de percentis por mês
    def pct(x, q): return float(np.percentile(x, q))
    summary = (
        df_sims.groupby("mes")["faturamento"]
        .agg(p10=lambda x: pct(x, 10),
             p60=lambda x: pct(x, 60),
             p90=lambda x: pct(x, 90))
        .reset_index()
    )

    # linha anual
    tot_por_sim = df_sims.groupby("sim")["faturamento"].sum().values
    annual = pd.DataFrame([{
        "mes": "annual",
        "p10": float(np.percentile(tot_por_sim, 10)),
        "p60": float(np.percentile(tot_por_sim, 60)),
        "p90": float(np.percentile(tot_por_sim, 90)),
    }])

    summary = pd.concat([summary, annual], ignore_index=True)
    summary.rename(columns={"mes": "metric"}, inplace=True)

    return df_sims, summary




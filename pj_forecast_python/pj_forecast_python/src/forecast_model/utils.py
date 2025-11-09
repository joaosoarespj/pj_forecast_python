import numpy as np
import pandas as pd
from typing import Dict, Iterable

# -----------------------
# Tempo / Sazonalidade
# -----------------------

def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de tempo padronizadas:
      - ano (YYYY)
      - mes (1..12)
      - trimestre (1..4)  -> TRIMESTRES de 3 meses (Q1=jan–mar, Q2=abr–jun, Q3=jul–set, Q4=out–dez)
    """
    out = df.copy()
    if "ano" not in out.columns:
        out["ano"] = out["data"].dt.year
    if "mes" not in out.columns:
        out["mes"] = out["data"].dt.month

    # TRIMESTRE CORRETO (3 meses): jan–mar=1, abr–jun=2, jul–set=3, out–dez=4
    out["trimestre"] = out["data"].dt.quarter

    return out



# -----------------------
# Estados (baixo/médio/alto)
# -----------------------

def classificar_estado_anual(
    df: pd.DataFrame,
    p_baixo: float = 0.40,
    p_medio: float = 0.60,
    p_alto: float = 0.80,
    col_ticket: str = "ticket"
) -> pd.DataFrame:
    """
    Classifica cada mês em 'baixo'/'medio'/'alto' POR ANO com base nos percentis do ticket daquele ano.
    Regras:
      - <= P35 -> baixo
      -  P35..P60 -> medio
      - >= P75 -> alto
      - (entre P60 e P75 fica em 'medio' para não criar quarto estado)
    Obs: inclui meses com ticket=0 na base, mas eles tendem a cair como 'baixo'.
    """
    df = df.copy()
    df["estado"] = "medio"

    for ano, sub in df.groupby("ano"):
        vals = sub[col_ticket].astype(float).values
        if len(vals) == 0:
            continue
        q_baixo = np.percentile(vals, p_baixo * 100)
        q_medio = np.percentile(vals, p_medio * 100)
        q_alto  = np.percentile(vals, p_alto  * 100)

        idx = sub.index
        t = sub[col_ticket].astype(float)

        df.loc[idx, "estado"] = np.where(
            t <= q_baixo, "baixo",
            np.where(t >= q_alto, "alto", "medio")
        )

    return df


# -----------------------
# Pesos por ano e percentis ponderados
# -----------------------

def pesos_por_ano(df: pd.DataFrame, anos_validos: Iterable[int]) -> Dict[int, float]:
    """
    Gera pesos que somam 1 dando maior peso ao ano mais recente.
    Ex.: se houver 2024, 2025, 2026 -> 0.25, 0.35, 0.40 (por exemplo).
    A curva abaixo é simples e pode ser ajustada.
    """
    anos = sorted(set(int(a) for a in anos_validos))
    if not anos:
        return {}
    # base linear crescente
    base = np.linspace(1.0, 1.0 + 0.2 * (len(anos) - 1), num=len(anos))
    w = base / base.sum()
    return {ano: float(w[i]) for i, ano in enumerate(anos)}


def percentil_ponderado_por_estado(
    df: pd.DataFrame,
    estado_col: str = "estado",
    ticket_col: str = "ticket",
    ano_min: int = 2024,
    percentile: float = 0.60,
) -> Dict[str, float]:
    """
    Calcula o 'ticket representativo' por estado (baixo/medio/alto) usando percentil ponderado
    considerando SOMENTE anos >= ano_min (ex.: 2024+). O ano mais recente recebe maior peso.
    """
    df = df.copy()
    df = df[df["ano"] >= ano_min].copy()
    if df.empty:
        # fallback: usa todo o período se não houver ano>=ano_min
        df = df.copy()

    anos = sorted(df["ano"].unique())
    w_ano = pesos_por_ano(df, anos)

    reps = {}
    for estado, sub in df.groupby(estado_col):
        # repete as linhas conforme peso relativo (amostragem por peso)
        blocos = []
        for ano, sub_ano in sub.groupby("ano"):
            peso = max(w_ano.get(int(ano), 0.0), 0.0)
            if peso == 0.0 or sub_ano.empty:
                continue
            # define multiplicador (quanto maior o peso, mais repetições)
            mult = int(round(100 * peso))
            if mult <= 0: mult = 1
            blocos.append(pd.concat([sub_ano]*mult, ignore_index=True))
        if not blocos:
            vals = sub[ticket_col].dropna().values
        else:
            vals = pd.concat(blocos, ignore_index=True)[ticket_col].dropna().values

        if len(vals) == 0:
            reps[estado] = 0.0
        else:
            reps[estado] = float(np.percentile(vals, percentile * 100))

    return reps



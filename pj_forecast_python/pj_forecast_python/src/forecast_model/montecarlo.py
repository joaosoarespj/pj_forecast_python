import numpy as np
import pandas as pd
from typing import Dict

def weighted_percentile(data, weights, percentile):
    """Calcula percentil ponderado."""
    data, weights = np.array(data), np.array(weights)
    sorter = np.argsort(data)
    data, weights = data[sorter], weights[sorter]
    cumsum = np.cumsum(weights)
    cutoff = percentile * cumsum[-1]
    return data[np.searchsorted(cumsum, cutoff)]

def state_ticket_representatives(df: pd.DataFrame, percentile: float = 0.60) -> Dict[str, float]:
    """
    Calcula o ticket típico (percentil) para cada estado,
    considerando apenas dados de 2024 em diante e dando
    mais peso aos anos mais recentes.
    """
    reps = {}
    df = df.copy()
    df["ano"] = df["data"].dt.year

    # usa só dados de 2024 pra frente
    df = df[df["ano"] >= 2024].copy()
    if df.empty:
        raise ValueError("Base histórica não tem dados de 2024 em diante.")

    # pesos crescentes por ano: 2024=1.0, 2025=1.5, 2026=2.0, etc.
    base = 2024
    df["peso"] = df["ano"].apply(lambda x: 1.0 + 0.5 * (x - base))

    for estado, sub in df.groupby("estado"):
        reps[estado] = float(weighted_percentile(sub["ticket"].dropna().values,
                                                 sub["peso"].values,
                                                 percentile))
    return reps


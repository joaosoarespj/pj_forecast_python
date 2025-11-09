import numpy as np
import pandas as pd
from typing import List, Dict

def estimate_transition_matrix(states: List[str], order: List[str]) -> pd.DataFrame:
    """Estimate 1-step transition matrix from a sequence of states.

    states: sequence like [Baixo, Medio, Alto, ...]

    order: list of unique states in fixed order, e.g. ["Baixo", "Médio", "Alto"]

    Returns a DataFrame P with rows=from, cols=to, each row sums to 1.

    """

    idx = {s:i for i,s in enumerate(order)}
    n = len(order)
    counts = np.zeros((n, n), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1.0
    # add Laplace smoothing to avoid zero rows
    counts += 1e-6
    P = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(P, index=order, columns=order)

def simulate_chain(P: pd.DataFrame, start_state: str, horizon: int, rng: np.random.Generator) -> List[str]:
    order = list(P.index)
    state = start_state
    path = [state]
    for _ in range(horizon-1):
        probs = P.loc[state].values
        nxt = rng.choice(order, p=probs)
        path.append(nxt)
        state = nxt
    return path
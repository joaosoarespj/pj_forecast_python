import os
import json
import pandas as pd

# ---------- Leitura ----------

def load_base_historica(path: str) -> pd.DataFrame:
    """
    Lê a base histórica. Requisitos:
      - colunas: data, vendas, ticket (opcionais: estado)
      - data como string no formato DD-MM-YYYY ou YYYY-MM-DD
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base histórica não encontrada em: {path}")

    # dayfirst=True para lidar com '01-02-2024' (01 de fevereiro)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "data" not in df.columns:
        raise ValueError("A base histórica precisa ter a coluna 'data'.")

    df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
    if df["data"].isna().any():
        # fallback: tenta no padrão iso
        df["data"] = pd.to_datetime(df["data"], errors="coerce", format="%Y-%m-%d")
    if df["data"].isna().any():
        linhas_ruins = df[df["data"].isna()]
        raise ValueError(f"Datas inválidas encontradas:\n{linhas_ruins}")

    # normaliza numéricos
    for c in ("vendas", "ticket"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month

    # garante colunas essenciais
    if "vendas" not in df.columns:
        df["vendas"] = 0.0
    if "ticket" not in df.columns:
        df["ticket"] = 0.0

    # higiene básica
    df["vendas"] = df["vendas"].clip(lower=0)
    df["ticket"] = df["ticket"].clip(lower=0)

    # padroniza 'estado' se existir
    if "estado" in df.columns:
        df["estado"] = df["estado"].astype(str).str.strip().str.lower()

    return df.sort_values("data").reset_index(drop=True)


# ---------- Escrita (com nomes únicos) ----------

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _unique_path(path: str) -> str:
    """
    Se 'path' já existir, devolve um novo com sufixo __02, __03, ...
    Ex.: summary_2025-11-09_101530.csv -> summary_2025-11-09_101530__02.csv
    """
    base, ext = os.path.splitext(path)
    i = 2
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}__{i:02d}{ext}"
        i += 1
    return candidate

def save_artifact_df(df: pd.DataFrame, path: str):
    ensure_dir(path)
    final = _unique_path(path)
    df.to_csv(final, index=False, encoding="utf-8-sig")
    print(f"✅ Arquivo salvo: {final}")

def save_artifact_json(obj: dict, path: str):
    ensure_dir(path)
    final = _unique_path(path)
    with open(final, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ Arquivo salvo: {final}")





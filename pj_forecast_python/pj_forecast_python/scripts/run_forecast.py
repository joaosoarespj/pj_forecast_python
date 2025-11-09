import os
from datetime import datetime
import numpy as np

from pj_forecast_python.src.forecast_model.data import (
    load_base_historica,
    save_artifact_df,
    save_artifact_json,
)
from pj_forecast_python.src.forecast_model.forecast import generate_forecast
from pj_forecast_python.src.forecast_model.utils import classificar_estado_anual

# Caminhos
BASE_PATH = os.getenv("BASE_HIST_PATH", "pj_forecast_python/data/base_historica.csv")
ART_DIR   = os.path.join("pj_forecast_python", "artifacts", "forecasts")
os.makedirs(ART_DIR, exist_ok=True)

def main():
    print("✅ Iniciando forecast...\n")

    # Seed dinâmica (variabilidade entre execuções; permite fixar FORECAST_SEED)
    user_seed = os.getenv("FORECAST_SEED")
    if user_seed:
        seed = int(user_seed)
        print(f"🎲 Usando seed definida pelo usuário: {seed}")
    else:
        seed = np.random.randint(0, 1_000_000)
        print(f"🎲 Seed aleatória gerada: {seed}")

    # 1) Carrega base
    df_hist = load_base_historica(BASE_PATH)
    # Se quiser forçar reclassificação (mesmo tendo coluna), comente/descomente abaixo:
    # df_hist = df_hist.drop(columns=["estado"], errors="ignore")
    df_hist = classificar_estado_anual(df_hist)

    # 2) Gera forecast
    df_forecast, df_summary = generate_forecast(
        df_hist,
        n_sims=5000,
        horizon_months=12,
        percentile_ticket=0.60,
        seed=seed
    )

    # 3) Salva artefatos com timestamp + proteção anti-sobrescrita
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # ex.: 2025-11-09_174052
    forecast_path = os.path.join(ART_DIR, f"forecast_{stamp}.csv")
    summary_path  = os.path.join(ART_DIR, f"summary_{stamp}.csv")
    meta_path     = os.path.join(ART_DIR, f"meta_{stamp}.json")

    save_artifact_df(df_forecast, forecast_path)
    save_artifact_df(df_summary, summary_path)
    save_artifact_json(
        {
            "generated_on": stamp,
            "seed": seed,
            "notes": "Forecast com Markov mensal + viés trimestral; vendas projetadas com sazonalidade + tendência; tickets P60 ponderados (>=2024). Nomes únicos com timestamp e fallback __02.",
        },
        meta_path
    )

    print("\n✅ Forecast concluído com sucesso!")
    print(f"📂 Resultados salvos em: {ART_DIR}")
    print(f"🔑 Seed usada: {seed}\n")

if __name__ == "__main__":
    main()











# PJ Forecast (Python)

Modelo de forecast em Python para tickets + cadeia de Markov, com atualizações trimestrais e controle de versões.

## Visão geral
- **Histórico**: base mensal com `vendas`, `ticket` e `estado` (Baixo/Médio/Alto).
- **Distribuições por estado**: usa percentil representativo (default P60) do ticket em cada estado.
- **Cadeia de Markov**: estimada a partir da sequência histórica de `estado` mensal.
- **Simulação**: gera trajetórias de estados e converte para faturamento = `ticket_do_estado` × `vendas_do_mês`.
- **Atualizações trimestrais**: salva versões do forecast (baseline de janeiro e re-forecasts Q1/Q2/Q3).
- **Dashboard**: `streamlit_app.py` para visualizar resultados e comparar versões.

## Estrutura
```
pj_forecast_python/
├── src/forecast_model/
│   ├── __init__.py
│   ├── data.py
│   ├── markov.py
│   ├── montecarlo.py
│   └── forecast.py
├── scripts/
│   └── run_forecast.py
├── streamlit_app.py
├── data/
│   ├── base_historica_template.csv
│   └── vendas_padrao_template.csv
├── artifacts/
│   └── forecasts/   # saídas versionadas (CSV/JSON)
├── requirements.txt
└── README.md
```

## Como rodar
1. Crie um virtualenv (opcional) e instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloque seus dados reais em `data/base_historica.csv` (ou edite o caminho em `scripts/run_forecast.py`).
3. Rode a simulação:
   ```bash
   python scripts/run_forecast.py
   ```
4. Veja o dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

## Dados esperados
### `base_historica.csv`
- `data`: primeiro dia de cada mês (YYYY-MM-01)
- `vendas`: inteiro com o multiplicador/qtde por mês (ex.: 1,2,3,4...)
- `ticket`: ticket médio observado no mês
- `estado`: Baixo | Médio | Alto

### `vendas_padrao.csv`
- `mes`: 1..12
- `vendas`: multiplicador ou qtde base por mês

## Versionamento de forecasts
Cada execução gera arquivos em `artifacts/forecasts`, por exemplo:
- `forecast_{YYYY-MM-DD}.csv`
- `summary_{YYYY-MM-DD}.json`

O baseline do início do ano pode ser mantido e comparado com re-forecasts trimestrais.

import pandas as pd
import os
df_total = pd.read_csv('ativos_filtrados.csv')

df_total['close'] = df_total['close'].astype(str).str.replace(',', '.', regex=False)
df_total['close'] = pd.to_numeric(df_total['close'], errors='coerce')
df_total = df_total.dropna(subset=['close'])
df_total['date'] = pd.to_datetime(df_total['date'], errors='coerce')
df_total = df_total.dropna(subset=['date'])

os.makedirs('resultados', exist_ok=True)

window_size = 20

for ticker in df_total['ticker'].unique():
    df_ticker = df_total[df_total['ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values('date')

    close = df_ticker['close'].tolist()
    rows = []

    for i in range(len(close) - window_size):
        janela = close[i:i + window_size]
        proximo = close[i + window_size]
        subiu = 'sim' if proximo > janela[-1] else 'não'
        rows.append(janela + [subiu])

    if rows:
        colunas = [f'Day {i+1}' for i in range(window_size)] + ['subiu?']
        df_resultado = pd.DataFrame(rows, columns=colunas)

        df_resultado.to_csv(f"resultados/resultado_{ticker.lower()}.csv", index=False)
        print(f"{ticker} processado com sucesso.")



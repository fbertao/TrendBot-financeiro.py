import pandas as pd

df = pd.read_excel('asle.xlsx')

print(df.head())
print(df.dtypes)

df['close'] = df['close'].astype(str).str.replace(',', '.', regex=False)

df['close'] = pd.to_numeric(df['close'], errors='coerce')

df = df.dropna(subset=['close'])

close = df['close'].tolist()

window_size = 20
rows = []

for i in range(len(close) - window_size):
    janela = close[i:i+window_size]
    proximo = close[i + window_size]
    subiu = 'Sim' if proximo > janela[-1] else 'Não'
    rows.append(janela + [subiu])

colunas = [f'Day {i+1}' for i in range(window_size)] + ['Subiu?']
df_resultado = pd.DataFrame(rows, columns=colunas)

print(df_resultado)


import duckdb

#excluindo tickers já usados
tickers_excluir = [
    'asle', 'asln', 'asmb', 'asnd',
    'aso', 'aspa', 'aspau', 'asrt'
]

tickers_sql = ", ".join(f"'{t}'" for t in tickers_excluir)

query = f"""
    SELECT *
    FROM 'infolimpioavanzadoTarget.csv'
    WHERE LOWER(Ticker) NOT IN ({tickers_sql})
"""
df_filtrado = duckdb.query(query).to_df()

df_filtrado.to_csv("ativos_filtrados.csv", index=False)

print("Linhas excluídas com sucesso. Novo arquivo salvo como 'ativos_filtrados.csv'.")

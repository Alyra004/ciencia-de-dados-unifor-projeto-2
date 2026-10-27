from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
caminho_base = BASE_DIR / 'dados' / 'raw' / 'MICRODADOS_ENEM_2023.csv'
if not caminho_base.exists():
    caminho_base = BASE_DIR / 'dados' / 'raw' / 'DADOS' / 'MICRODADOS_ENEM_2023.csv'
pasta_splits = BASE_DIR / 'dados' / 'splits'
pasta_docs = BASE_DIR / 'docs'

# 1. Carregar a base completa e contabilizar o diagnóstico
df = pd.read_csv(caminho_base, sep=';', encoding='latin1')

qtd_inicial = len(df)
print(f"Quantidade inicial de registros: {qtd_inicial}")  # Deve imprimir 3933955

# 2. Definir o alvo
alvo = 'NU_NOTA_MT'

df_modelagem = df.dropna(subset=[alvo])

qtd_exclusoes = qtd_inicial - len(df_modelagem)
qtd_final = len(df_modelagem)

print(f"Registros excluídos por nota ausente no alvo '{alvo}': {qtd_exclusoes}")
print(f"Total de participantes utilizados para a modelagem: {qtd_final}")

# 3. Dividir os dados em 80% para desenvolvimento e 20% para teste
df_dev, df_teste = train_test_split(df_modelagem, test_size=0.2, random_state=42)

qtd_dev = len(df_dev)
qtd_teste = len(df_teste)

# 4. Salvar os identificadores (splits) no repositório
pasta_splits.mkdir(parents=True, exist_ok=True)

df_dev[['NU_INSCRICAO']].to_csv(pasta_splits / 'desenvolvimento_ids.csv', index=False)
df_teste[['NU_INSCRICAO']].to_csv(pasta_splits / 'teste_ids.csv', index=False)

print("Identificadores das partições salvos com sucesso em dados/splits/!")

# 5. Salvar relatório de diagnóstico em arquivo de texto em docs/
pasta_docs.mkdir(parents=True, exist_ok=True)
arquivo_diagnostico = pasta_docs / 'diagnostico_triagem.txt'

relatorio_diagnostico = f"""================================================================================
RELATÓRIO DE DIAGNÓSTICO E TRIAGEM - MICRODADOS ENEM 2023
================================================================================

1. BASE NACIONAL
----------------
- Arquivo utilizado: {caminho_base.relative_to(BASE_DIR)}
- Quantidade inicial de registros: {qtd_inicial:,}

2. TRIAGEM DO ALVO
------------------
- Variável alvo: {alvo}
- Registros excluídos por ausência de nota: {qtd_exclusoes:,} ({(qtd_exclusoes / qtd_inicial) * 100:.2f}%)
- Total de participantes selecionados para modelagem: {qtd_final:,} ({(qtd_final / qtd_inicial) * 100:.2f}%)

3. PARTICIONAMENTO DOS DADOS (SPLITS - RANDOM_STATE=42)
-------------------------------------------------------
- Partição de Desenvolvimento (80%): {qtd_dev:,} registros ({(qtd_dev / qtd_final) * 100:.2f}%)
  -> Identificadores salvos em: dados/splits/desenvolvimento_ids.csv
- Partição de Teste (20% - Isolada): {qtd_teste:,} registros ({(qtd_teste / qtd_final) * 100:.2f}%)
  -> Identificadores salvos em: dados/splits/teste_ids.csv

================================================================================
"""

with open(arquivo_diagnostico, 'w', encoding='utf-8') as f:
    f.write(relatorio_diagnostico)

print(f"Relatório de diagnóstico salvo com sucesso em: docs/diagnostico_triagem.txt")
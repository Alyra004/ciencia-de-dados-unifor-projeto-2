from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
caminho_base = BASE_DIR / 'dados' / 'raw' / 'MICRODADOS_ENEM_2023.csv'
if not caminho_base.exists():
    caminho_base = BASE_DIR / 'dados' / 'raw' / 'DADOS' / 'MICRODADOS_ENEM_2023.csv'
pasta_splits = BASE_DIR / 'dados' / 'splits'

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

# 4. Salvar os identificadores (splits) no repositório
pasta_splits.mkdir(parents=True, exist_ok=True)

df_dev[['NU_INSCRICAO']].to_csv(pasta_splits / 'desenvolvimento_ids.csv', index=False)
df_teste[['NU_INSCRICAO']].to_csv(pasta_splits / 'teste_ids.csv', index=False)

print("Identificadores das partições salvos com sucesso em dados/splits/!")
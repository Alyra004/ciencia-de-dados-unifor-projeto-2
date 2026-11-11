"""Recria (ou confere) a partição 80/20 e as dobras da validação cruzada.

Uso, a partir da raiz do projeto, depois de rodar o notebook 01 (que grava dados/processed/ids_elegiveis.csv):

    python src/particao.py           # confere que os arquivos de dados/splits/ são iguais aos gerados com a semente 42
    python src/particao.py --gravar  # sobrescreve dados/splits/

- ids_elegiveis.csv: participantes com nota de Matemática observada (2.692.427), na ordem do arquivo bruto.
- Desenvolvimento / teste: train_test_split(ids, test_size=0.2, random_state=42).
- Dobras: KFold(n_splits=3, shuffle=True, random_state=42) sobre os participantes do desenvolvimento, na ordem de
  ids_elegiveis.csv (a dobra 1, 2 ou 3 de cada um é a dobra em que ele é validação).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ids = pd.read_csv("dados/processed/ids_elegiveis.csv")["NU_INSCRICAO"]
dev, teste = train_test_split(ids, test_size=0.2, random_state=42)
dev_na_ordem = ids[ids.isin(set(dev))]
dobra = np.zeros(len(dev_na_ordem), dtype=int)
for k, (_, val) in enumerate(KFold(n_splits=3, shuffle=True, random_state=42).split(dev_na_ordem), 1):
    dobra[val] = k

saidas = {"desenvolvimento_ids.csv": dev.to_frame(), "teste_ids.csv": teste.to_frame(), "dobras_cv.csv": pd.DataFrame({"dobra": dobra})}
pasta = Path("dados/splits")
if "--gravar" in sys.argv:
    for nome, df in saidas.items():
        df.to_csv(pasta / nome, index=False)
    print("gravado em", pasta)
else:
    for nome, df in saidas.items():
        salvo = pd.read_csv(pasta / nome)
        igual = (set(salvo.iloc[:, 0]) == set(df.iloc[:, 0])) if "ids" in nome else (salvo["dobra"].to_numpy() == dobra).all()
        print(f"{nome}: {'igual' if igual else 'DIFERENTE'}")

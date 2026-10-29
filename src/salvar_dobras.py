"""Salva em dados/splits/dobras_cv.csv a dobra (1, 2 ou 3) de cada participante do desenvolvimento.

São as dobras da validação cruzada usada em todos os experimentos: KFold(n_splits=3, shuffle=True, random_state=42)
aplicado às linhas do desenvolvimento NA ORDEM do arquivo numérico (a mesma ordem que os scripts e notebooks de
modelagem usam). A dobra indicada é aquela em que o participante fica na VALIDAÇÃO; nas outras duas ele é treino.
O teste não é lido. Se o arquivo já existe, o script só confere que a divisão reproduzida é a mesma, sem sobrescrever
(use --sobrescrever para regravar).

Uso:
    python src/salvar_dobras.py [--sobrescrever]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import CAMINHO_DESENVOLVIMENTO_NUMERICO, CAMINHO_IDS_DESENVOLVIMENTO, CHAVE_ID, DIR_SPLITS, SEMENTE

CAMINHO_DOBRAS = DIR_SPLITS / 'dobras_cv.csv'


def calcular_dobras():
    """DataFrame (NU_INSCRICAO, dobra) na ordem do arquivo numérico de desenvolvimento."""
    ids = pd.read_csv(CAMINHO_DESENVOLVIMENTO_NUMERICO, usecols=[CHAVE_ID], dtype={CHAVE_ID: 'int64'})[CHAVE_ID]
    dobra = np.zeros(len(ids), dtype='int8')
    for numero, (_, validacao) in enumerate(KFold(n_splits=3, shuffle=True, random_state=SEMENTE).split(ids), start=1):
        dobra[validacao] = numero
    return pd.DataFrame({CHAVE_ID: ids, 'dobra': dobra})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sobrescrever', action='store_true', help='regrava o arquivo mesmo que já exista')
    args = ap.parse_args()

    dobras = calcular_dobras()
    ids_dev = pd.read_csv(CAMINHO_IDS_DESENVOLVIMENTO, dtype={CHAVE_ID: 'int64'})[CHAVE_ID]
    assert set(dobras[CHAVE_ID]) == set(ids_dev), 'os IDs das dobras não coincidem com dados/splits/desenvolvimento_ids.csv'
    assert dobras[CHAVE_ID].is_unique, 'há NU_INSCRICAO repetido no desenvolvimento'
    print('Participantes por dobra (validação):', dobras['dobra'].value_counts().sort_index().to_dict())
    print('Mesma ordem do arquivo de IDs de desenvolvimento:', bool(np.array_equal(dobras[CHAVE_ID].to_numpy(), ids_dev.to_numpy())))

    if CAMINHO_DOBRAS.exists() and not args.sobrescrever:
        salvo = pd.read_csv(CAMINHO_DOBRAS, dtype={CHAVE_ID: 'int64'})
        igual = salvo.equals(dobras)
        print(f'{CAMINHO_DOBRAS.name} já existe; divisão reproduzida {"é igual" if igual else "DIFERE"} (nada foi sobrescrito).')
        return
    DIR_SPLITS.mkdir(parents=True, exist_ok=True)
    dobras.to_csv(CAMINHO_DOBRAS, index=False)
    print(f'Salvo: {CAMINHO_DOBRAS}')


if __name__ == '__main__':
    main()

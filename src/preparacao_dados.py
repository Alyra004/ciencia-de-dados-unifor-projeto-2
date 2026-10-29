"""Triagem do alvo e partição desenvolvimento/teste (80/20, random_state=42).

Conta os registros da base nacional, retira os sem nota de Matemática (sem imputar
nem converter em zero), separa 80% para desenvolvimento e 20% para teste e guarda
só os identificadores em dados/splits/. Só lê NU_INSCRICAO e NU_NOTA_MT.

Se as partições já existem, o script apenas confere se a divisão reproduzida é a
mesma e não sobrescreve nada (use --sobrescrever para regravar).

Uso:
    python src/preparacao_dados.py [--sobrescrever] [--sem-relatorio]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (ALVO, CAMINHO_IDS_DESENVOLVIMENTO, CAMINHO_IDS_TESTE, CHAVE_ID,
                    DIR_DOCS, DIR_SPLITS, SEMENTE, BASE_DIR, ler_csv_bruto,
                    localizar_csv_bruto)


def triar_e_particionar(caminho_base):
    """Devolve (registros iniciais, ids de desenvolvimento, ids de teste)."""
    df = ler_csv_bruto(caminho_base, usecols=[CHAVE_ID, ALVO])
    qtd_inicial = len(df)
    elegiveis = df.dropna(subset=[ALVO])
    dev, teste = train_test_split(elegiveis, test_size=0.2, random_state=SEMENTE)
    return qtd_inicial, len(elegiveis), dev[CHAVE_ID].reset_index(drop=True), teste[CHAVE_ID].reset_index(drop=True)


def conferir_particoes_salvas(ids_dev, ids_teste):
    """Compara a divisão reproduzida com os arquivos de dados/splits/."""
    salvo_dev = pd.read_csv(CAMINHO_IDS_DESENVOLVIMENTO)[CHAVE_ID]
    salvo_teste = pd.read_csv(CAMINHO_IDS_TESTE)[CHAVE_ID]
    mesma_ordem = np.array_equal(salvo_dev, ids_dev) and np.array_equal(salvo_teste, ids_teste)
    mesmo_conjunto = set(salvo_dev) == set(ids_dev) and set(salvo_teste) == set(ids_teste)
    return mesma_ordem, mesmo_conjunto


def montar_relatorio(caminho_base, qtd_inicial, qtd_final, qtd_dev, qtd_teste):
    exclusoes = qtd_inicial - qtd_final
    return f"""================================================================================
RELATÓRIO DE DIAGNÓSTICO E TRIAGEM - MICRODADOS ENEM 2023
================================================================================

1. BASE NACIONAL
----------------
- Arquivo utilizado: {caminho_base.relative_to(BASE_DIR).as_posix()}
- Quantidade inicial de registros: {qtd_inicial:,}

2. TRIAGEM DO ALVO
------------------
- Variável alvo: {ALVO}
- Registros excluídos por ausência de nota: {exclusoes:,} ({exclusoes / qtd_inicial * 100:.2f}%)
- Total de participantes selecionados para modelagem: {qtd_final:,} ({qtd_final / qtd_inicial * 100:.2f}%)

3. PARTICIONAMENTO DOS DADOS (SPLITS - RANDOM_STATE={SEMENTE})
-------------------------------------------------------
- Partição de Desenvolvimento (80%): {qtd_dev:,} registros ({qtd_dev / qtd_final * 100:.2f}%)
  -> Identificadores salvos em: dados/splits/desenvolvimento_ids.csv
- Partição de Teste (20% - Isolada): {qtd_teste:,} registros ({qtd_teste / qtd_final * 100:.2f}%)
  -> Identificadores salvos em: dados/splits/teste_ids.csv

================================================================================
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sobrescrever', action='store_true', help='regrava os IDs mesmo que já existam')
    ap.add_argument('--sem-relatorio', action='store_true', help='não regrava docs/txt/diagnostico_triagem.txt')
    args = ap.parse_args()

    caminho_base = localizar_csv_bruto()
    qtd_inicial, qtd_final, ids_dev, ids_teste = triar_e_particionar(caminho_base)
    print(f'Quantidade inicial de registros: {qtd_inicial:,}')
    print(f'Registros excluídos por nota ausente em {ALVO}: {qtd_inicial - qtd_final:,}')
    print(f'Participantes elegíveis para a modelagem: {qtd_final:,}')
    print(f'Desenvolvimento: {len(ids_dev):,} | Teste: {len(ids_teste):,}')

    ja_existem = CAMINHO_IDS_DESENVOLVIMENTO.exists() and CAMINHO_IDS_TESTE.exists()
    if ja_existem and not args.sobrescrever:
        mesma_ordem, mesmo_conjunto = conferir_particoes_salvas(ids_dev, ids_teste)
        if mesma_ordem:
            print('Partições salvas em dados/splits/ conferem com a divisão reproduzida (mesma ordem). Nada foi regravado.')
        elif mesmo_conjunto:
            print('Partições salvas contêm os mesmos participantes, em outra ordem. Nada foi regravado.')
        else:
            raise SystemExit('As partições salvas NÃO correspondem à divisão reproduzida. '
                             'Investigue antes de continuar (ou use --sobrescrever, ciente de que invalida os resultados).')
    else:
        DIR_SPLITS.mkdir(parents=True, exist_ok=True)
        ids_dev.to_csv(CAMINHO_IDS_DESENVOLVIMENTO, index=False)
        ids_teste.to_csv(CAMINHO_IDS_TESTE, index=False)
        print('Identificadores das partições salvos em dados/splits/.')

    if not args.sem_relatorio:
        destino_pasta = DIR_DOCS / 'txt'
        destino_pasta.mkdir(parents=True, exist_ok=True)
        destino = DIR_DOCS / 'txt' / 'diagnostico_triagem.txt'
        destino.write_text(montar_relatorio(caminho_base, qtd_inicial, qtd_final, len(ids_dev), len(ids_teste)),
                           encoding='utf-8')
        print(f'Relatório salvo em {destino.relative_to(BASE_DIR).as_posix()}')


if __name__ == '__main__':
    main()

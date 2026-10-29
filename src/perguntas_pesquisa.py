"""Contagens das perguntas de pesquisa registradas na seção 1.9 do documento de decisões.

Saídas em resultados/tabelas/ (ou --saida/tabelas):
  proporcao_nao_sei.csv  participantes do DESENVOLVIMENTO que responderam "não sei" (H em Q001/Q002,
                         F em Q003/Q004), com a nota média de quem respondeu e dos demais. O teste não é usado.
  cor_raca_sexo.csv      cor/raça × sexo entre os participantes com nota de Matemática (descritivo).
  viuvos_cor_sexo.csv    viúvos por cor/raça e sexo, mesma base (descritivo).

As duas últimas usam a base elegível inteira (todos com NU_NOTA_MT), como as figuras descritivas do
notebook de análise exploratória; não servem para escolher features nem modelos.

Uso:
    python src/perguntas_pesquisa.py [--saida DIR]
"""
import argparse
from pathlib import Path

import pandas as pd

from config import (ALVO, CAMINHO_IDS_DESENVOLVIMENTO, CHAVE_ID, DIR_RESULTADOS, TAMANHO_BLOCO,
                    ler_csv_bruto, localizar_csv_bruto)

COLUNAS = [CHAVE_ID, ALVO, 'Q001', 'Q002', 'Q003', 'Q004', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_SEXO']
COR_RACA = {0: 'Não declarado', 1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indígena', 6: 'Não dispõe da informação'}
SEXO = {'F': 'Feminino', 'M': 'Masculino'}
VIUVO = 4   # TP_ESTADO_CIVIL

# "Não sei" é a letra H em Q001/Q002 (escolaridade) e F em Q003/Q004 (ocupação).
SINALIZADORES = {
    'Q001 = H (não sei, pai ou responsável)': lambda d: d['Q001'] == 'H',
    'Q002 = H (não sei, mãe ou responsável)': lambda d: d['Q002'] == 'H',
    'Q001 ou Q002 = H (escolaridade)': lambda d: (d['Q001'] == 'H') | (d['Q002'] == 'H'),
    'Q001 e Q002 = H (os dois)': lambda d: (d['Q001'] == 'H') & (d['Q002'] == 'H'),
    'Q003 = F (não sei, ocupação do pai)': lambda d: d['Q003'] == 'F',
    'Q004 = F (não sei, ocupação da mãe)': lambda d: d['Q004'] == 'F',
    'Q003 ou Q004 = F (ocupação)': lambda d: (d['Q003'] == 'F') | (d['Q004'] == 'F'),
    'Qualquer "não sei" em Q001 a Q004': lambda d: ((d['Q001'] == 'H') | (d['Q002'] == 'H')
                                                    | (d['Q003'] == 'F') | (d['Q004'] == 'F')),
}


def contar(caminho_csv, ids_dev):
    """Uma passada pelo CSV bruto: acumula os sinalizadores (desenvolvimento) e os cruzamentos (base elegível)."""
    acum = {nome: {'n': 0, 'soma': 0.0} for nome in SINALIZADORES}
    total_dev = {'n': 0, 'soma': 0.0}
    cruzamentos = []
    tipos = {c: 'string' for c in ('Q001', 'Q002', 'Q003', 'Q004', 'TP_SEXO')}
    for bloco in ler_csv_bruto(caminho_csv, usecols=COLUNAS, dtype=tipos, chunksize=TAMANHO_BLOCO):
        bloco = bloco.dropna(subset=[ALVO])
        cruzamentos.append(bloco.groupby(['TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_SEXO']).size())
        dev = bloco[bloco[CHAVE_ID].isin(ids_dev)]
        total_dev['n'] += len(dev)
        total_dev['soma'] += float(dev[ALVO].sum())
        for nome, regra in SINALIZADORES.items():
            marcados = dev[regra(dev).fillna(False)]
            acum[nome]['n'] += len(marcados)
            acum[nome]['soma'] += float(marcados[ALVO].sum())
    return acum, total_dev, pd.concat(cruzamentos).groupby(level=[0, 1, 2]).sum()


def tabela_nao_sei(acum, total_dev):
    linhas = []
    for nome, a in acum.items():
        demais_n, demais_soma = total_dev['n'] - a['n'], total_dev['soma'] - a['soma']
        media_sim, media_nao = a['soma'] / a['n'], demais_soma / demais_n
        linhas.append({'pergunta': nome, 'registros': a['n'], 'proporcao_pct': round(a['n'] / total_dev['n'] * 100, 2),
                       'nota_media_nao_sei': round(media_sim, 1), 'nota_media_demais': round(media_nao, 1),
                       'diferenca_pontos': round(media_sim - media_nao, 1)})
    return pd.DataFrame(linhas), total_dev['n']


def tabelas_cruzamento(cruzamento):
    df = cruzamento.rename('registros').reset_index()
    df['cor_raca'] = df['TP_COR_RACA'].map(COR_RACA)
    df['sexo'] = df['TP_SEXO'].map(SEXO)
    cor_sexo = df.groupby(['TP_COR_RACA', 'cor_raca', 'sexo'], as_index=False)['registros'].sum()
    cor_sexo['proporcao_na_cor_raca_pct'] = (cor_sexo['registros'] / cor_sexo.groupby('cor_raca')['registros']
                                             .transform('sum') * 100).round(1)
    viuvos = df[df['TP_ESTADO_CIVIL'] == VIUVO].groupby(['TP_COR_RACA', 'cor_raca', 'sexo'], as_index=False)['registros'].sum()
    viuvos['proporcao_dos_viuvos_pct'] = (viuvos['registros'] / viuvos['registros'].sum() * 100).round(1)
    return cor_sexo.drop(columns='TP_COR_RACA'), viuvos.drop(columns='TP_COR_RACA')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    args = ap.parse_args()

    ids_dev = set(pd.read_csv(CAMINHO_IDS_DESENVOLVIMENTO, dtype={CHAVE_ID: 'int64'})[CHAVE_ID])
    acum, total_dev, cruzamento = contar(localizar_csv_bruto(), ids_dev)
    nao_sei, n_dev = tabela_nao_sei(acum, total_dev)
    cor_sexo, viuvos = tabelas_cruzamento(cruzamento)

    pasta = args.saida / 'tabelas'
    pasta.mkdir(parents=True, exist_ok=True)
    nao_sei.to_csv(pasta / 'proporcao_nao_sei.csv', index=False)
    cor_sexo.to_csv(pasta / 'cor_raca_sexo.csv', index=False)
    viuvos.to_csv(pasta / 'viuvos_cor_sexo.csv', index=False)

    print(f'"Não sei" no desenvolvimento (n={n_dev:,}):\n{nao_sei.to_string(index=False)}')
    print(f'\nViúvos: {int(viuvos["registros"].sum()):,} | cor/raça × sexo: {int(cor_sexo["registros"].sum()):,} registros')
    print(f'Saídas em {pasta}')


if __name__ == '__main__':
    main()

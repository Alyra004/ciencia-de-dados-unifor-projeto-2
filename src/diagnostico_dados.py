"""Diagnóstico da base nacional: ausências, alvo, treineiros, outliers e duplicatas.

Uma passagem em blocos pela base completa (3.933.955 registros). Registros sem nota
entram no diagnóstico, mas não na regressão. Os pontos sinalizados pelo critério
1,5 × IQR são apenas investigados: nada é removido. Tabelas salvas em
resultados/tabelas/.

Uso:
    python src/diagnostico_dados.py
"""
import numpy as np
import pandas as pd

from config import (ALVO, CHAVE_ID, COLUNAS_DIAGNOSTICO, DIR_RESULTADOS, FEATURES_CANDIDATAS,
                    TAMANHO_BLOCO, ler_csv_bruto, localizar_csv_bruto)


def varrer_base(caminho_base):
    """Conta linhas, nota ausente, treineiros e ausências por coluna; guarda as notas observadas."""
    cont = dict(linhas=0, id_ausente=0, treineiros=0, treineiros_com_nota=0, treineiros_sem_nota=0)
    ausentes = {c: 0 for c in FEATURES_CANDIDATAS}
    ausentes_com_nota = {c: 0 for c in FEATURES_CANDIDATAS}
    notas = []
    leitor = ler_csv_bruto(caminho_base, usecols=COLUNAS_DIAGNOSTICO, chunksize=TAMANHO_BLOCO,
                           dtype={c: 'string' for c in [CHAVE_ID, *FEATURES_CANDIDATAS]})
    for n, bloco in enumerate(leitor, start=1):
        cont['linhas'] += len(bloco)
        observado = bloco[ALVO].notna()
        cont['id_ausente'] += int(bloco[CHAVE_ID].isna().sum())
        treineiro = bloco['IN_TREINEIRO'].eq('1').fillna(False)
        cont['treineiros'] += int(treineiro.sum())
        cont['treineiros_com_nota'] += int((treineiro & observado).sum())
        cont['treineiros_sem_nota'] += int((treineiro & ~observado).sum())
        for coluna in FEATURES_CANDIDATAS:
            falta = bloco[coluna].isna()
            ausentes[coluna] += int(falta.sum())
            ausentes_com_nota[coluna] += int((falta & observado).sum())
        notas.append(pd.to_numeric(bloco.loc[observado, ALVO], errors='coerce').dropna().to_numpy(dtype='float64'))
        if n % 10 == 0:
            print(f'  blocos lidos: {n} | linhas: {cont["linhas"]:,}')
    return cont, ausentes, ausentes_com_nota, np.concatenate(notas)


def resumir_outliers(notas):
    """Limites de 1,5 × IQR como sinalizador exploratório (nada é removido)."""
    q1, mediana, q3 = np.quantile(notas, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    abaixo, acima = notas < inf, notas > sup
    return pd.DataFrame([
        ('Notas observadas', len(notas)), ('Nota mínima', notas.min()), ('Nota máxima', notas.max()),
        ('Q1', q1), ('Mediana', mediana), ('Q3', q3), ('IQR', iqr),
        ('Limite inferior IQR', inf), ('Limite superior IQR', sup),
        ('Candidatos abaixo do limite IQR', int(abaixo.sum())),
        ('Desses, notas zero (válidas)', int((notas[abaixo] == 0).sum())),
        ('Candidatos acima do limite IQR', int(acima.sum())),
        ('Candidatos fora da escala 0-1000', int(((notas < 0) | (notas > 1000)).sum())),
    ], columns=['medida', 'valor'])


def verificar_ids(caminho_base):
    ids = ler_csv_bruto(caminho_base, usecols=[CHAVE_ID])[CHAVE_ID]
    return pd.DataFrame([{
        'registros': len(ids), 'IDs ausentes': int(ids.isna().sum()),
        'IDs duplicados após a primeira ocorrência': int(ids.duplicated().sum()),
        'IDs distintos': int(ids.nunique(dropna=True)),
    }])


def main():
    caminho_base = localizar_csv_bruto()
    print(f'Base: {caminho_base}')
    cont, ausentes, ausentes_com_nota, notas = varrer_base(caminho_base)
    com_nota = len(notas)

    resumo = pd.DataFrame([
        ('Registros na base bruta', cont['linhas']),
        ('Registros sem NU_NOTA_MT (fora da regressão)', cont['linhas'] - com_nota),
        ('Registros com NU_NOTA_MT (elegíveis)', com_nota),
        ('Treineiros na base bruta', cont['treineiros']),
        ('Treineiros elegíveis mantidos (nota observada)', cont['treineiros_com_nota']),
        ('Treineiros sem nota-alvo (não entram na regressão)', cont['treineiros_sem_nota']),
        ('NU_INSCRICAO ausente', cont['id_ausente']),
    ], columns=['indicador', 'quantidade'])

    tabela_ausencias = pd.DataFrame([{
        'feature': c,
        'ausentes_base': ausentes[c],
        'percentual_ausente_base': round(ausentes[c] / cont['linhas'] * 100, 2),
        'ausentes_com_nota_alvo': ausentes_com_nota[c],
        'percentual_ausente_com_nota': round(ausentes_com_nota[c] / com_nota * 100, 2),
    } for c in FEATURES_CANDIDATAS]).sort_values('percentual_ausente_base', ascending=False, ignore_index=True)

    outliers = resumir_outliers(notas)
    ids = verificar_ids(caminho_base)

    pasta = DIR_RESULTADOS / 'tabelas'
    pasta.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(pasta / 'resumo_limpeza.csv', index=False)
    tabela_ausencias.to_csv(pasta / 'ausencias_features.csv', index=False)
    outliers.to_csv(pasta / 'resumo_outliers.csv', index=False)

    print('\nResumo da base:\n', resumo.to_string(index=False))
    print('\nFeatures candidatas com ausência real:\n',
          tabela_ausencias.loc[tabela_ausencias['ausentes_base'] > 0].to_string(index=False))
    print('\nOutliers da nota (sinalizador IQR; nenhum ponto é removido):\n', outliers.to_string(index=False))
    print('\nIdentificador:\n', ids.to_string(index=False))
    print(f'\nTabelas salvas em {pasta}')


if __name__ == '__main__':
    main()

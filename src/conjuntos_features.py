"""Conjuntos de features dos experimentos e o Pipeline de cada um, iguais para todos os modelos.

  A  originais tratadas: as 138 colunas numéricas (33 features originais codificadas)
  B  A + 5 derivadas calculadas a partir das próprias colunas (143 colunas)
  D  B com os grupos redundantes fundidos (5 colunas viram 2; 140 colunas). A fusão fica dentro do Pipeline,
     depois da imputação e antes do modelo, e aprende a média e o desvio só no treino de cada dobra.

As derivadas usam as mesmas fórmulas do notebook 02 (tabela de features derivadas); nada é aprendido dos
dados nelas, então não há vazamento. O uso é:
    X_conjunto, nomes_conjunto = montar_conjunto('B', X, nomes)
    pipeline = criar_pipeline('B', Ridge(alpha=1.0, solver='lsqr'), nomes_conjunto)
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fusao_features import FusaoPorGrupos, indices_dos_grupos, nomes_apos_fusao

DERIVADAS = ['RENDA_PER_CAPITA', 'ESC_MAX_RESP', 'INDICE_DIGITAL', 'MORADORES_POR_QUARTO', 'RENDA_X_ESC_MAX']
CONJUNTOS = {
    'A': 'Originais tratadas (138 colunas)',
    'B': 'A + 5 derivadas (143 colunas)',
    'D': 'B com os grupos redundantes fundidos (140 colunas)',
}


def _coluna(X, nomes, nome):
    """Uma coluna da matriz esparsa como vetor denso (NaN preservado)."""
    return X[:, nomes.index(nome)].toarray().ravel().astype('float32')


def _razao(numerador, denominador):
    """Divisão com denominador inválido (menor que 1 ou ausente) explicitado como ausente, e não como zero."""
    denominador = np.where(denominador >= 1, denominador, np.nan)
    return (numerador / denominador).astype('float32')


def calcular_derivadas(X, nomes):
    """As 5 derivadas, na ordem de DERIVADAS, como matriz densa (n × 5)."""
    renda = _coluna(X, nomes, 'Q006_RENDA_SM_ESTIMADA')
    moradores = _coluna(X, nomes, 'Q005_MORADORES')
    quartos = _coluna(X, nomes, 'Q009_CONTAGEM_ESTIMADA')
    esc_max = np.fmax(_coluna(X, nomes, 'Q001_NIVEL_ESCOLARIDADE'), _coluna(X, nomes, 'Q002_NIVEL_ESCOLARIDADE'))
    tem_computador = np.nan_to_num(_coluna(X, nomes, 'Q024_CONTAGEM_ESTIMADA')) > 0
    tem_internet = np.nan_to_num(_coluna(X, nomes, 'Q025_BINARIA')) == 1
    return np.column_stack([
        _razao(renda, moradores),
        esc_max,
        tem_computador.astype('float32') + tem_internet.astype('float32'),
        _razao(moradores, quartos),
        (renda * esc_max).astype('float32'),
    ]).astype('float32')


def montar_conjunto(conjunto, X, nomes):
    """Devolve (matriz, nomes) que entram no Pipeline. Em D a matriz é a de B: a fusão acontece dentro do Pipeline."""
    if conjunto == 'A':
        return X, list(nomes)
    if conjunto in ('B', 'D'):
        derivadas = sparse.csr_matrix(calcular_derivadas(X, nomes))
        return sparse.hstack([X, derivadas], format='csr', dtype=np.float32), list(nomes) + DERIVADAS
    raise KeyError(f'Conjunto desconhecido: {conjunto}. Opções: {list(CONJUNTOS)}')


def colunas_finais(conjunto, nomes_matriz):
    """Nomes das colunas que o modelo de fato recebe (depois da fusão, no caso de D)."""
    return nomes_apos_fusao(nomes_matriz) if conjunto == 'D' else list(nomes_matriz)


def criar_pipeline(conjunto, estimador, nomes_matriz, preparar=True):
    """Imputação pela mediana, fusão (só em D) e padronização esparsa, depois o estimador.

    `preparar=False` devolve só o estimador (o baseline pela média não usa features, então não precisa de preparo).
    """
    if not preparar:
        return estimador
    etapas = [('imputacao', SimpleImputer(strategy='median', keep_empty_features=True))]
    if conjunto == 'D':
        etapas.append(('fusao', FusaoPorGrupos(indices_dos_grupos(nomes_matriz))))
    etapas += [('padronizacao', StandardScaler(with_mean=False)), ('modelo', estimador)]
    return Pipeline(etapas)


def avaliar_nos_conjuntos(nome, criar_estimador, parametros, conjuntos, matrizes, y, cv, scoring, preparar=True):
    """Roda um modelo em cada conjunto (validação cruzada), imprime os parâmetros e devolve a tabela de métricas.

    `criar_estimador` é uma função sem argumentos que devolve o estimador já com os parâmetros; `matrizes` mapeia
    cada conjunto para (matriz, nomes), como devolve `montar_conjunto`.
    """
    print(f'Modelo: {nome}')
    print(f'Parâmetros declarados: {parametros}')
    print(f'Todos os parâmetros do estimador: {criar_estimador().get_params()}')
    linhas = []
    for conjunto in conjuntos:
        X_conj, nomes_matriz = matrizes[conjunto]
        pipeline = criar_pipeline(conjunto, criar_estimador(), nomes_matriz, preparar=preparar)
        r = cross_validate(pipeline, X_conj, y, cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
        rmse, mae, r2 = -r['test_rmse'], -r['test_mae'], r['test_r2']
        linhas.append({'conjunto': conjunto, 'colunas': len(colunas_finais(conjunto, nomes_matriz)),
                       'RMSE': f'{rmse.mean():.3f} ± {rmse.std(ddof=1):.3f}', 'MAE': f'{mae.mean():.3f} ± {mae.std(ddof=1):.3f}',
                       'R²': f'{r2.mean():.4f} ± {r2.std(ddof=1):.4f}', 'tempo_ajuste_s': round(float(np.sum(r['fit_time'])), 1)})
    return pd.DataFrame(linhas)

"""Registro dos experimentos em resultados/experimentos.csv.

Cada linha guarda hipótese, features, preparação, modelo, parâmetros, conjunto
avaliado, métricas por dobra, média, desvio padrão, tempo e conclusão. Reexecutar
um experimento substitui a linha de mesmo `id_experimento`; os demais são mantidos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import DIR_RESULTADOS

COLUNAS = [
    'id_experimento', 'hipotese', 'features', 'preparacao', 'modelo', 'parametros', 'conjunto_avaliado',
    'rmse_dobra1', 'rmse_dobra2', 'rmse_dobra3', 'rmse_media', 'rmse_dp',
    'mae_dobra1', 'mae_dobra2', 'mae_dobra3', 'mae_media', 'mae_dp',
    'r2_dobra1', 'r2_dobra2', 'r2_dobra3', 'r2_media', 'r2_dp',
    'tempo_ajuste_s', 'conclusao',
]


def linha_de_cv(id_experimento, hipotese, features, preparacao, modelo, parametros, resultado, conclusao,
                conjunto='desenvolvimento (validação cruzada, 3 dobras)'):
    """Monta a linha a partir do dicionário devolvido por sklearn.model_selection.cross_validate."""
    rmse, mae, r2 = -resultado['test_rmse'], -resultado['test_mae'], resultado['test_r2']
    linha = {'id_experimento': id_experimento, 'hipotese': hipotese, 'features': features,
             'preparacao': preparacao, 'modelo': modelo, 'parametros': parametros,
             'conjunto_avaliado': conjunto, 'tempo_ajuste_s': float(np.sum(resultado['fit_time'])),
             'conclusao': conclusao}
    for nome, valores in (('rmse', rmse), ('mae', mae), ('r2', r2)):
        for i, v in enumerate(valores, start=1):
            linha[f'{nome}_dobra{i}'] = float(v)
        linha[f'{nome}_media'] = float(np.mean(valores))
        linha[f'{nome}_dp'] = float(np.std(valores, ddof=1))
    return linha


def registrar(linhas, pasta_saida=DIR_RESULTADOS):
    """Grava/atualiza as linhas em <pasta_saida>/experimentos.csv e devolve o DataFrame completo."""
    destino = Path(pasta_saida) / 'experimentos.csv'
    destino.parent.mkdir(parents=True, exist_ok=True)
    novas = pd.DataFrame(linhas, columns=COLUNAS)
    if destino.exists():
        atual = pd.read_csv(destino)
        atual = atual[~atual['id_experimento'].isin(novas['id_experimento'])]
        novas = pd.concat([atual, novas], ignore_index=True)
    novas.to_csv(destino, index=False)
    return novas

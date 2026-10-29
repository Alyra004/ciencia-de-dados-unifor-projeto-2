"""Gráfico previsão × observado e erros do Ridge em cada conjunto de features (A, B e D).

Mesmo gráfico do notebook 04 (src/grafico_regressao.py), com todos os participantes do desenvolvimento e previsões fora da
dobra (KFold(3, shuffle=True, random_state=42)); o teste não é lido. Os conjuntos estão em src/conjuntos_features.py.
Saídas em resultados/figuras/: regressao_ridge_conjunto_A.png, ..._B.png e ..._D.png.

Uso:
    python src/figuras_regressao_conjuntos.py [--saida DIR]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from config import DIR_RESULTADOS
from conjuntos_features import CONJUNTOS, colunas_finais, criar_pipeline, montar_conjunto
from grafico_regressao import metricas, plotar_previsao_vs_observado, previsoes_fora_da_dobra
from modelagem_acompanhamento import carregar_desenvolvimento_esparso

PARAMETROS = {'alpha': 1.0, 'solver': 'lsqr'}
PREPARACAO = {'A': 'imputação pela mediana + padronização esparsa (StandardScaler, with_mean=False), reajustadas no treino de cada dobra',
              'B': 'derivadas calculadas antes; imputação pela mediana + padronização esparsa, reajustadas no treino de cada dobra',
              'D': 'derivadas calculadas antes; imputação, fusão de 5 colunas em 2 e padronização esparsa, reajustadas no treino de cada dobra'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    args = ap.parse_args()
    pasta = args.saida / 'figuras'
    pasta.mkdir(parents=True, exist_ok=True)

    X, y, nomes = carregar_desenvolvimento_esparso()
    y = np.asarray(y, dtype=np.float32)
    zeros = y == 0
    for conjunto, descricao in CONJUNTOS.items():
        X_conj, nomes_matriz = montar_conjunto(conjunto, X, nomes)
        n_colunas = len(colunas_finais(conjunto, nomes_matriz))
        modelo = criar_pipeline(conjunto, Ridge(**PARAMETROS), nomes_matriz)
        previsao = previsoes_fora_da_dobra(modelo, X_conj, y)
        m = metricas(y, previsao)
        fig = plotar_previsao_vs_observado(y, previsao, f'Ridge (alpha=1), conjunto {conjunto}', PARAMETROS,
                                           f'conjunto {conjunto}: {descricao}', m, PREPARACAO[conjunto])
        fig.savefig(pasta / f'regressao_ridge_conjunto_{conjunto}.png', dpi=150)
        print(f'Conjunto {conjunto} ({n_colunas} colunas): RMSE {m["rmse"]:.3f} | MAE {m["mae"]:.3f} | R² {m["r2"]:.4f} | '
              f'reta: previsto = {m["intercepto"]:.1f} + {m["inclinacao"]:.3f} × observado | viés {m["vies"]:+.3f} | '
              f'previsões de {previsao.min():.0f} a {previsao.max():.0f} (1%-99%: {np.percentile(previsao, 1):.0f}-'
              f'{np.percentile(previsao, 99):.0f}) | previsão média para as {int(zeros.sum()):,} notas zero: {previsao[zeros].mean():.0f}',
              flush=True)
    print(f'Figuras em {pasta}')


if __name__ == '__main__':
    main()

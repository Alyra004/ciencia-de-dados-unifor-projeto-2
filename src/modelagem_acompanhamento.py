"""Modelagem preliminar do acompanhamento: baseline pela média × Ridge.

Compara DummyRegressor(strategy='mean') e Ridge(alpha=1) por validação cruzada de três
dobras (KFold(3, shuffle=True, random_state=42)) no DESENVOLVIMENTO completo. O arquivo de
teste não é referenciado em nenhum ponto. Imputação (mediana) e padronização ficam dentro
do Pipeline, reajustadas no treino de cada dobra. Métricas na escala original da nota.

Saídas: resultados/experimentos.csv, resultados/tabelas/acompanhamento_cv_*.csv e
resultados/figuras/acompanhamento_baseline_ridge.png.

Uso:
    python src/modelagem_acompanhamento.py [--saida DIR] [--limite-linhas N]
"""
import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALVO, CAMINHO_DESENVOLVIMENTO_NUMERICO, CHAVE_ID, DIR_RESULTADOS, SEMENTE, TAMANHO_BLOCO
from registro_experimentos import linha_de_cv, registrar

SCORING = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
DESCRICAO_FEATURES = ('33 features originais (perfil/trajetória escolar + Q001-Q025), codificadas em '
                      'colunas numéricas; sem as 21 features criadas')


def carregar_desenvolvimento_esparso(limite_linhas=None):
    """Lê o desenvolvimento numérico em blocos e devolve (X esparsa, y, nomes das colunas)."""
    caminho = CAMINHO_DESENVOLVIMENTO_NUMERICO
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não encontrado. Execute 'python src/higienizacao.py' primeiro.")
    colunas = pd.read_csv(caminho, nrows=0).columns.tolist()
    if ALVO not in colunas or CHAVE_ID not in colunas:
        raise KeyError('A base de desenvolvimento precisa conter NU_NOTA_MT e NU_INSCRICAO.')
    features = [c for c in colunas if c not in {CHAVE_ID, ALVO}]

    blocos_x, blocos_y, linhas = [], [], 0
    for n, bloco in enumerate(pd.read_csv(caminho, usecols=[*features, ALVO], dtype=np.float32,
                                          chunksize=TAMANHO_BLOCO, nrows=limite_linhas), start=1):
        y = bloco[ALVO].to_numpy(dtype=np.float32, copy=True)
        if np.isnan(y).any():
            raise ValueError('Há nota-alvo ausente na partição de desenvolvimento.')
        x = csr_matrix(bloco[features].to_numpy(dtype=np.float32, copy=False))
        x.eliminate_zeros()
        blocos_x.append(x)
        blocos_y.append(y)
        linhas += len(bloco)
        if n % 5 == 0:
            print(f'  blocos lidos: {n} | linhas: {linhas:,}')
    X = vstack(blocos_x, format='csr', dtype=np.float32)
    y = np.concatenate(blocos_y).astype(np.float32, copy=False)
    del blocos_x, blocos_y
    gc.collect()
    return X, y, features


def criar_ridge(alpha=1.0):
    """Pipeline do Ridge: imputação e padronização são reajustadas no treino de cada ajuste."""
    return Pipeline([
        ('imputacao', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('padronizacao', StandardScaler(with_mean=False)),
        ('modelo', Ridge(alpha=alpha, solver='lsqr')),
    ])


def criar_modelos():
    return {'Baseline (média)': DummyRegressor(strategy='mean'), 'Ridge (alpha=1)': criar_ridge(1.0)}


def validar(X, y):
    cv = KFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    resultados = {}
    for nome, modelo in criar_modelos().items():
        print(f'Validação cruzada: {nome}')
        resultados[nome] = cross_validate(modelo, X, y, cv=cv, scoring=SCORING, n_jobs=1,
                                          return_train_score=False, error_score='raise')
    return resultados


def tabelas(resultados):
    por_dobra, resumo = [], []
    for nome, r in resultados.items():
        for i in range(3):
            por_dobra.append({'modelo': nome, 'dobra': i + 1, 'RMSE': -r['test_rmse'][i],
                              'MAE': -r['test_mae'][i], 'R2': r['test_r2'][i], 'tempo_ajuste_s': r['fit_time'][i]})
        resumo.append({'modelo': nome,
                       'RMSE_medio': -r['test_rmse'].mean(), 'RMSE_dp': r['test_rmse'].std(ddof=1),
                       'MAE_medio': -r['test_mae'].mean(), 'MAE_dp': r['test_mae'].std(ddof=1),
                       'R2_medio': r['test_r2'].mean(), 'R2_dp': r['test_r2'].std(ddof=1),
                       'tempo_ajuste_total_s': r['fit_time'].sum()})
    return pd.DataFrame(por_dobra), pd.DataFrame(resumo).sort_values('RMSE_medio', ignore_index=True)


def plotar(df_dobras, df_resumo, n_linhas, destino):
    """Três painéis: RMSE, MAE e R² (média ± desvio entre dobras; pontos = dobras)."""
    cores = {'Baseline (média)': '#8c8c8c', 'Ridge (alpha=1)': '#1f77b4'}
    modelos = list(cores)
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (col, titulo, resumo_col, dp_col) in zip(eixos, [
            ('RMSE', 'RMSE (pontos da nota)', 'RMSE_medio', 'RMSE_dp'),
            ('MAE', 'MAE (pontos da nota)', 'MAE_medio', 'MAE_dp'),
            ('R2', 'R²', 'R2_medio', 'R2_dp')]):
        for i, m in enumerate(modelos):
            linha = df_resumo.loc[df_resumo['modelo'] == m].iloc[0]
            ax.bar(i, linha[resumo_col], yerr=linha[dp_col], color=cores[m], capsize=6, width=0.55)
            pontos = df_dobras.loc[df_dobras['modelo'] == m, col]
            ax.scatter([i] * len(pontos), pontos, color='black', s=14, zorder=3)
            valor = linha[resumo_col]
            rotulo = f'{abs(valor) if abs(valor) < 0.005 else valor:.2f}' if col == 'R2' else f'{valor:.1f}'
            ax.annotate(rotulo, (i, valor), xytext=(0, 9), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(['Baseline\n(média)', 'Ridge\n(alpha=1)'])
        ax.set_title(titulo)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Validação cruzada (3 dobras) no desenvolvimento, n={n_linhas:,}: baseline × Ridge',
                 fontsize=12, fontweight='bold')
    fig.text(0.01, 0.005, 'Fonte: Inep, ENEM 2023. Barras: média; barra de erro: desvio padrão; pontos: dobras.', fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    ap.add_argument('--limite-linhas', type=int, default=None,
                    help='lê só as primeiras N linhas (teste rápido; não usar para reportar resultados)')
    args = ap.parse_args()
    plt.switch_backend('Agg')  # só grava arquivos; não abre janela

    print(f'Arquivo de desenvolvimento: {CAMINHO_DESENVOLVIMENTO_NUMERICO} (o teste não é carregado)')
    X, y, features = carregar_desenvolvimento_esparso(args.limite_linhas)
    print(f'Registros: {X.shape[0]:,} | features: {X.shape[1]} | nota média: {y.mean():.2f}')

    resultados = validar(X, y)
    df_dobras, df_resumo = tabelas(resultados)
    print('\nPor dobra:\n', df_dobras.to_string(index=False))
    print('\nResumo (média e desvio entre as 3 dobras):\n', df_resumo.to_string(index=False))

    pasta_tab, pasta_fig = args.saida / 'tabelas', args.saida / 'figuras'
    pasta_tab.mkdir(parents=True, exist_ok=True)
    pasta_fig.mkdir(parents=True, exist_ok=True)
    df_dobras.to_csv(pasta_tab / 'acompanhamento_cv_por_dobra.csv', index=False)
    df_resumo.to_csv(pasta_tab / 'acompanhamento_cv_resumo.csv', index=False)
    plotar(df_dobras, df_resumo, X.shape[0], pasta_fig / 'acompanhamento_baseline_ridge.png')

    rb = df_resumo.set_index('modelo')
    ganho = (1 - rb.loc['Ridge (alpha=1)', 'RMSE_medio'] / rb.loc['Baseline (média)', 'RMSE_medio']) * 100
    base_linha = dict(features=DESCRICAO_FEATURES,
                      preparacao='imputação pela mediana + padronização esparsa dentro do Pipeline, por dobra')
    registrar([
        linha_de_cv('acomp_baseline_media', 'Referência: prever a média do treino de cada dobra para todos',
                    modelo='DummyRegressor', parametros='strategy=mean', resultado=resultados['Baseline (média)'],
                    conclusao='RMSE próximo do desvio padrão da nota; R² ~ 0 por construção.', **base_linha),
        linha_de_cv('acomp_ridge_alpha1', 'O perfil socioeconômico e escolar explica parte da nota (relação linear)',
                    modelo='Ridge', parametros='alpha=1.0; solver=lsqr', resultado=resultados['Ridge (alpha=1)'],
                    conclusao=f'RMSE {ganho:.1f}% menor que o baseline, com pouca variação entre dobras; '
                              'resultado preliminar, sem busca de hiperparâmetros.', **base_linha),
    ], pasta_saida=args.saida)
    print(f'\nRidge reduz o RMSE em {ganho:.1f}% frente ao baseline.')
    print(f'Saídas em {args.saida} (experimentos.csv, tabelas/, figuras/). Teste intocado.')


if __name__ == '__main__':
    main()

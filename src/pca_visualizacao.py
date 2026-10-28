"""Visualização exploratória por PCA (2D e 3D) das features de ENEM 2023.

Uso apenas diagnóstico, sobre uma amostra documentada da partição de
DESENVOLVIMENTO (o teste não é lido). A nota é usada só para colorir os pontos;
não entra no PCA. Não substitui o experimento de PCA da modelagem, que deve ser
reajustado dentro de cada dobra da validação cruzada.

Uso:
    python src/pca_visualizacao.py [--entrada CSV] [--amostra 0.01] [--bloco numericas|todas]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engenharia_features import FEATURES_CRIADAS, criar_features

BASE_DIR = Path(__file__).resolve().parent.parent
ENTRADA_PADRAO = BASE_DIR / 'dados' / 'processed' / 'desenvolvimento_higienizado.csv'
PASTA_FIGURAS = BASE_DIR / 'resultados' / 'figuras'
ALVO = 'NU_NOTA_MT'
SEMENTE = 42
CATEGORICAS = (
    ['TP_SEXO', 'SG_UF_PROVA', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_NACIONALIDADE',
     'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_LINGUA']
    + [f'Q{i:03d}' for i in range(1, 26)]
)


def ler_amostra(caminho, fracao):
    """Amostra aleatória por blocos (memória constante), semente fixa."""
    rng = np.random.RandomState(SEMENTE)
    partes = []
    for bloco in pd.read_csv(caminho, chunksize=500_000, dtype={c: str for c in CATEGORICAS}):
        partes.append(bloco.sample(frac=fracao, random_state=rng))
    return pd.concat(partes, ignore_index=True)


def montar_matriz(df, bloco):
    df = criar_features(df)
    numericas = list(FEATURES_CRIADAS) + ['TP_FAIXA_ETARIA', 'IN_TREINEIRO']
    prep = [('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('esc', StandardScaler())]), numericas)]
    if bloco == 'todas':
        prep.append(('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01,
                                          sparse_output=False), CATEGORICAS))
    return ColumnTransformer(prep).fit_transform(df)


def plotar(Z, nota, var, sufixo):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    kw = dict(c=nota, cmap='viridis', s=4, alpha=0.35, linewidths=0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], **kw)
    ax.set(xlabel=f'PC1 ({var[0]:.1%})', ylabel=f'PC2 ({var[1]:.1%})',
           title=f'PCA 2D - desenvolvimento (n={len(Z):,})')
    fig.colorbar(sc, label=ALVO)
    fig.text(0.01, 0.01, 'Fonte: Inep, ENEM 2023', fontsize=7)
    fig.savefig(PASTA_FIGURAS / f'pca_2d_{sufixo}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], **kw)
    ax.set(xlabel=f'PC1 ({var[0]:.1%})', ylabel=f'PC2 ({var[1]:.1%})', zlabel=f'PC3 ({var[2]:.1%})',
           title=f'PCA 3D - desenvolvimento (n={len(Z):,})')
    fig.colorbar(sc, label=ALVO, shrink=0.6)
    fig.text(0.01, 0.01, 'Fonte: Inep, ENEM 2023', fontsize=7)
    fig.savefig(PASTA_FIGURAS / f'pca_3d_{sufixo}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plotar_variancia(X, sufixo):
    pca = PCA(random_state=SEMENTE).fit(X)
    acum = np.cumsum(pca.explained_variance_ratio_)
    n80 = int(np.searchsorted(acum, 0.80) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(acum) + 1), acum, marker='o', ms=3)
    ax.axhline(0.80, ls='--', c='gray')
    ax.axvline(n80, ls=':', c='gray')
    ax.set(xlabel='Componentes', ylabel='Variância explicada acumulada',
           title=f'PCA: {n80} componentes atingem 80% (amostra do desenvolvimento)')
    fig.text(0.01, 0.01, 'Fonte: Inep, ENEM 2023', fontsize=7)
    fig.savefig(PASTA_FIGURAS / f'pca_variancia_{sufixo}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return n80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', type=Path, default=ENTRADA_PADRAO)
    ap.add_argument('--amostra', type=float, default=0.01, help='fração do desenvolvimento (padrão 1%%)')
    ap.add_argument('--bloco', choices=['numericas', 'todas'], default='numericas')
    args = ap.parse_args()

    df = ler_amostra(args.entrada, args.amostra)
    print(f'Amostra do desenvolvimento: {len(df):,} linhas (fração={args.amostra}, semente={SEMENTE})')

    X = montar_matriz(df, args.bloco)
    pca = PCA(n_components=3, random_state=SEMENTE).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_
    print(f'Variância explicada PC1-3: {np.round(var, 4)} (total {var.sum():.1%})')

    plotar(Z, df[ALVO].to_numpy(), var, args.bloco)
    n80 = plotar_variancia(X, args.bloco)
    print(f'Componentes para 80% da variância: {n80} de {X.shape[1]}')
    print(f'Figuras salvas em {PASTA_FIGURAS}')


if __name__ == '__main__':
    main()

"""PCA exploratória (2D e 3D) das 138 colunas numéricas, com TODOS os participantes do desenvolvimento.

Uso só diagnóstico e visual: o PCA é ajustado no desenvolvimento completo (o teste não é lido) e a nota serve
apenas para colorir os pontos; ela não entra no PCA. Não é o experimento de PCA da modelagem, que deve ser
reajustado dentro de cada dobra da validação cruzada.

Método: imputação pela mediana e padronização (z-score) das 138 colunas, depois autovalores e autovetores da
matriz de covariância (138 × 138), acumulada em blocos. Todos os participantes são projetados; nenhuma amostra
entra no cálculo. Só o desenho 3D usa uma amostra aleatória de pontos (2 milhões de pontos não cabem num gráfico 3D).

Saídas em resultados/ (ou --saida):
  figuras/pca_2d_desenvolvimento.png       2D: densidade e nota média por hexágono (todos os participantes)
  figuras/pca_3d_desenvolvimento.png       3D (amostra só para o desenho) e variância explicada
  tabelas/pca_variancia_explicada.csv      variância explicada por componente
  tabelas/pca_cargas_desenvolvimento.csv   peso de cada coluna em PC1, PC2 e PC3

Uso:
    python src/pca_dados_completos.py [--saida DIR] [--amostra-3d N]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DIR_RESULTADOS, SEMENTE
from modelagem_acompanhamento import carregar_desenvolvimento_esparso

BLOCO = 200_000
COR_NOTA = dict(cmap='viridis', vmin=300, vmax=800)   # cor limitada a 300-800; notas fora ficam nas cores extremas


def pca_por_covariancia(Xs):
    """Autovalores/autovetores da covariância de Xs (esparsa, já padronizada), acumulada em blocos de linhas."""
    n, p = Xs.shape
    soma = np.zeros(p)
    g = np.zeros((p, p))
    for i in range(0, n, BLOCO):
        bloco = Xs[i:i + BLOCO].astype(np.float64)
        soma += np.asarray(bloco.sum(axis=0)).ravel()
        g += (bloco.T @ bloco).toarray()
    media = soma / n
    cov = (g - n * np.outer(media, media)) / (n - 1)
    autovalores, autovetores = np.linalg.eigh(cov)
    ordem = np.argsort(autovalores)[::-1]
    return autovalores[ordem], autovetores[:, ordem], media, np.diag(cov).copy()


def projetar(Xs, componentes, media):
    """Pontuação de todos os participantes nas componentes dadas (centraliza pela média das colunas)."""
    deslocamento = media @ componentes
    partes = [np.asarray(Xs[i:i + BLOCO] @ componentes) - deslocamento for i in range(0, Xs.shape[0], BLOCO)]
    return np.vstack(partes).astype(np.float32)


def _br(v, casas=1):
    return f'{v:.{casas}f}'.replace('.', ',')


def plotar_2d(pc, y, razao, destino):
    """Esquerda: densidade (nº de participantes por hexágono, escala log). Direita: nota média por hexágono."""
    x1, x2 = pc[:, 0], pc[:, 1]
    lim = [np.percentile(v, [0.01, 99.99]) for v in (x1, x2)]
    extent = (lim[0][0], lim[0][1], lim[1][0], lim[1][1])
    fora = int(((x1 < extent[0]) | (x1 > extent[1]) | (x2 < extent[2]) | (x2 > extent[3])).sum())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.4))
    fig.subplots_adjust(left=0.065, right=0.955, top=0.8, bottom=0.12, wspace=0.3)
    hb1 = ax1.hexbin(x1, x2, gridsize=90, extent=extent, mincnt=1, cmap='Blues', norm=LogNorm())
    cb1 = fig.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label('participantes por hexágono (escala log)', fontsize=8.5)
    ax1.set_title('Densidade: onde estão os participantes', fontsize=11, fontweight='bold')
    hb2 = ax2.hexbin(x1, x2, C=y, reduce_C_function=np.mean, gridsize=90, extent=extent, mincnt=30, **COR_NOTA)
    cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label('nota média (hexágonos com ≥ 30 participantes)', fontsize=8.5)
    ax2.set_title('Nota média de Matemática', fontsize=11, fontweight='bold')
    for ax in (ax1, ax2):
        ax.set_xlabel(f'PC1 ({_br(razao[0] * 100)}% da variância)')
        ax.set_ylabel(f'PC2 ({_br(razao[1] * 100)}% da variância)')
        ax.grid(alpha=0.25)
    fig.suptitle('PCA em 2 dimensões: todos os participantes do desenvolvimento', fontsize=14, fontweight='bold', y=0.985)
    n_txt, fora_txt = f'{len(y):,}'.replace(',', '.'), f'{fora:,}'.replace(',', '.')
    fig.text(0.065, 0.935, f'n = {n_txt} participantes, 138 colunas padronizadas. As duas primeiras componentes explicam '
             f'{_br((razao[0] + razao[1]) * 100)}% da variância.\nEixos até os percentis 0,01 e 99,99 ({fora_txt} participantes ficam fora do quadro).',
             ha='left', va='top', fontsize=8.8, color='#333333', linespacing=1.4)
    fig.text(0.01, 0.005, 'Fonte: Inep, ENEM 2023. A nota só colore os pontos; não entra no PCA.', fontsize=7.5)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return fora


def plotar_3d_e_variancia(pc, y, razao, n_amostra, destino):
    """Esquerda: 3D de uma amostra aleatória (só para o desenho). Direita: variância explicada por componente."""
    rng = np.random.RandomState(SEMENTE)
    idx = rng.choice(len(y), size=n_amostra, replace=False)
    fig = plt.figure(figsize=(11.6, 5.6))
    ax3 = fig.add_axes([0.0, 0.2, 0.52, 0.58], projection='3d')
    pontos = ax3.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2], c=y[idx], s=2.2, alpha=0.45, rasterized=True, **COR_NOTA)
    ax3.set_xlabel(f'PC1 ({_br(razao[0] * 100)}%)', fontsize=8.5, labelpad=-2)
    ax3.set_ylabel(f'PC2 ({_br(razao[1] * 100)}%)', fontsize=8.5, labelpad=-2)
    ax3.set_zlabel(f'PC3 ({_br(razao[2] * 100)}%)', fontsize=8.5, labelpad=2)
    ax3.tick_params(labelsize=7, pad=-2)
    for eixo in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        eixo.set_major_locator(MaxNLocator(5))   # poucos rótulos para não amontoar
    ax3.view_init(elev=22, azim=-58)
    ax3.set_title(f'3 dimensões (amostra de {n_amostra:,} pontos)'.replace(',', '.'), fontsize=11, fontweight='bold')
    cb = fig.colorbar(pontos, cax=fig.add_axes([0.12, 0.1, 0.3, 0.022]), orientation='horizontal')
    cb.set_label('nota de Matemática', fontsize=8.5)

    ax = fig.add_axes([0.6, 0.12, 0.37, 0.58])
    k = 15
    ax.bar(np.arange(1, k + 1), razao[:k] * 100, color=['#d62728'] * 3 + ['#4C78A8'] * (k - 3))
    acumulada = np.cumsum(razao) * 100
    ax.plot(np.arange(1, k + 1), acumulada[:k], color='#ff7f0e', marker='o', ms=4, lw=1.8, label='acumulada')
    for n_comp in (2, 3):
        ax.annotate(f'{n_comp} comp.: {_br(acumulada[n_comp - 1])}%', (n_comp, acumulada[n_comp - 1]), xytext=(9, -15),
                    textcoords='offset points', fontsize=8.3, fontweight='bold')
    ax.set_xlabel('componente principal')
    ax.set_ylabel('variância explicada (%)')
    ax.set_title('Variância explicada', fontsize=11, fontweight='bold')
    ax.set_xticks(range(1, k + 1))
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='center right', fontsize=8.5, frameon=False)
    fig.suptitle('PCA em 3 dimensões e variância explicada', fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.02, 0.93, f'O PCA usa os {len(y):,} participantes do desenvolvimento'.replace(',', '.')
             + f'; só o desenho 3D usa uma amostra aleatória de {n_amostra:,}.'.replace(',', '.'),
             ha='left', va='top', fontsize=8.8, color='#333333')
    fig.text(0.01, 0.005, 'Fonte: Inep, ENEM 2023. A nota só colore os pontos; não entra no PCA.', fontsize=7.5)
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    ap.add_argument('--amostra-3d', type=int, default=30_000, help='pontos desenhados no gráfico 3D (o PCA usa todos)')
    args = ap.parse_args()

    X, y, nomes = carregar_desenvolvimento_esparso()
    Xi = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(X)
    Xs = sparse.csr_matrix(StandardScaler(with_mean=False).fit_transform(Xi))
    del Xi
    autovalores, autovetores, media, variancias = pca_por_covariancia(Xs)
    razao = autovalores / autovalores.sum()
    pc = projetar(Xs, autovetores[:, :3], media)
    y = np.asarray(y, dtype=np.float32)

    pasta_fig, pasta_tab = args.saida / 'figuras', args.saida / 'tabelas'
    pasta_fig.mkdir(parents=True, exist_ok=True)
    pasta_tab.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'componente': np.arange(1, len(razao) + 1), 'variancia_explicada_pct': razao * 100,
                  'acumulada_pct': np.cumsum(razao) * 100}).to_csv(pasta_tab / 'pca_variancia_explicada.csv', index=False)
    cargas = pd.DataFrame(autovetores[:, :3], index=nomes, columns=['PC1', 'PC2', 'PC3']).round(4)
    cargas.to_csv(pasta_tab / 'pca_cargas_desenvolvimento.csv', index_label='coluna')

    fora = plotar_2d(pc, y, razao, pasta_fig / 'pca_2d_desenvolvimento.png')
    plotar_3d_e_variancia(pc, y, razao, args.amostra_3d, pasta_fig / 'pca_3d_desenvolvimento.png')

    acumulada = np.cumsum(razao) * 100
    print(f'Participantes no PCA: {len(y):,} | colunas: {Xs.shape[1]} | colunas sem variação (constantes): {int((variancias < 1e-12).sum())} | autovalores nulos: {int((autovalores < 1e-12).sum())}')
    print('Variância explicada (%):', ', '.join(f'PC{i + 1} {razao[i] * 100:.2f}' for i in range(5)),
          f'| acumulada 2: {acumulada[1]:.2f}, 3: {acumulada[2]:.2f}')
    for alvo in (50, 80, 90):
        print(f'  componentes para {alvo}% da variância: {int(np.searchsorted(acumulada, alvo) + 1)}')
    for i in range(3):
        r = np.corrcoef(pc[:, i].astype(np.float64), y.astype(np.float64))[0, 1]
        mais = cargas.iloc[:, i].abs().sort_values(ascending=False).head(6).index
        print(f'  PC{i + 1}: correlação com a nota = {r:+.3f} | maiores pesos: '
              + ', '.join(f'{n} ({cargas.loc[n].iloc[i]:+.2f})' for n in mais))
    print(f'Participantes fora do quadro 2D: {fora:,}')
    print(f'Saídas em {args.saida}')


if __name__ == '__main__':
    main()

"""Análise exploratória do desenvolvimento: distribuição do alvo e categorias raras.

Usa só a partição de DESENVOLVIMENTO (o teste nunca é lido): lê a base limpa em
blocos e mantém apenas as linhas cujo ID está em dados/splits/desenvolvimento_ids.csv.

Saídas em resultados/figuras/ (ou --saida):
  distribuicao_alvo.png    histograma de NU_NOTA_MT com as notas zero em destaque
  analise_categorias.png   frequência de Q006, TP_COR_RACA, TP_ESCOLA e TP_ESTADO_CIVIL
e resultados/tabelas/categorias_raras.csv.

Uso:
    python src/analise_exploratoria.py [--saida DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from config import (ALVO, CAMINHO_BASE_LIMPA, CAMINHO_IDS_DESENVOLVIMENTO, CHAVE_ID,
                    DIR_RESULTADOS, TAMANHO_BLOCO)

FEATURES_PARA_ANALISE = ['Q006', 'TP_COR_RACA', 'TP_ESCOLA', 'TP_ESTADO_CIVIL']
LIMIAR_RARIDADE_PCT = 1.5


def carregar_desenvolvimento(colunas):
    """Lê da base limpa só as linhas do desenvolvimento (códigos originais, como texto)."""
    for caminho in (CAMINHO_BASE_LIMPA, CAMINHO_IDS_DESENVOLVIMENTO):
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não encontrado. Execute 'python src/preparacao_dados.py' "
                                    "e 'python src/higienizacao.py' primeiro.")
    ids_dev = set(pd.read_csv(CAMINHO_IDS_DESENVOLVIMENTO, dtype={CHAVE_ID: 'int64'})[CHAVE_ID])
    partes = []
    for bloco in pd.read_csv(CAMINHO_BASE_LIMPA, usecols=[CHAVE_ID, *colunas], chunksize=TAMANHO_BLOCO,
                             dtype={c: str for c in colunas if c != ALVO}):
        partes.append(bloco[bloco[CHAVE_ID].isin(ids_dev)])
    return pd.concat(partes, ignore_index=True)


def plotar_distribuicao_alvo(df, destino):
    nota = df[ALVO]
    media, mediana = nota.mean(), nota.median()
    zeros = int((nota == 0).sum())
    pct_zeros = zeros / len(df) * 100
    print('=' * 60)
    print(f'DIAGNÓSTICO DA VARIÁVEL ALVO ({ALVO}) - desenvolvimento, n={len(df):,}')
    print(f'Média: {media:.2f} | Mediana: {mediana:.2f} | Desvio-padrão: {nota.std():.2f}')
    print(f'Mínimo: {nota.min():.2f} | Máximo: {nota.max():.2f}')
    print(f'Notas exatamente zero: {zeros:,} ({pct_zeros:.3f}%)')
    print('=' * 60)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df[df[ALVO] > 0], x=ALVO, bins=60, color='#1f77b4', kde=True, stat='count', ax=ax,
                 label='Notas contínuas (TRI > 0)')
    if zeros:
        ax.bar(x=0, height=zeros, width=18, color='#d62728', edgecolor='black',
               label=f'Notas = 0.0 ({zeros:,} candidatos)', zorder=5)
        ax.annotate(f'Concentração de Zeros:\n{zeros:,} alunos ({pct_zeros:.2f}%)', xy=(0, zeros),
                    xytext=(85, zeros * 0.9 if zeros > 1000 else zeros + 8000),
                    arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
                    fontsize=10, fontweight='bold', color='#d62728',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff2f2', edgecolor='#d62728', alpha=0.9))
    ax.axvline(media, color='#2ca02c', linestyle='--', linewidth=1.5, label=f'Média ({media:.1f})')
    ax.axvline(mediana, color='#ff7f0e', linestyle=':', linewidth=1.5, label=f'Mediana ({mediana:.1f})')
    ax.set_title('Distribuição da Nota de Matemática (ENEM 2023) - Destaque para Zeros', fontsize=13, pad=12)
    ax.set_xlabel('Nota de Matemática (NU_NOTA_MT)', fontsize=11)
    ax.set_ylabel('Quantidade de Inscritos', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    fig.text(0.01, 0.005, 'Fonte: Inep, ENEM 2023 (desenvolvimento)', fontsize=7)
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    print(f'Gráfico salvo: {destino}')


def plotar_categorias_raras(df, destino_fig, destino_tabela):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    linhas_raras = []
    total = len(df)
    for ax, col in zip(axes.flatten(), FEATURES_PARA_ANALISE):
        contagens = df[col].fillna('(ausente)').astype(str).value_counts().sort_index()
        pct = contagens / total * 100
        paleta = ['#e67e22' if p < LIMIAR_RARIDADE_PCT else '#34495e' for p in pct]
        sns.barplot(x=contagens.index, y=contagens.values, hue=contagens.index, palette=paleta,
                    legend=False, ax=ax)
        ax.set_title(f'Distribuição da Feature: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Categoria ({col})', fontsize=10)
        ax.set_ylabel('Frequência Absoluta', fontsize=10)
        for barra, p in zip(ax.patches, pct):
            ax.text(barra.get_x() + barra.get_width() / 2.0, barra.get_height() + total * 0.005, f'{p:.1f}%',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold' if p < LIMIAR_RARIDADE_PCT else 'normal',
                    color='#d35400' if p < LIMIAR_RARIDADE_PCT else '#2c3e50')
        for categoria, p in pct[pct < LIMIAR_RARIDADE_PCT].items():
            linhas_raras.append({'variavel': col, 'categoria': categoria,
                                 'frequencia_absoluta': int(contagens[categoria]), 'proporcao_pct': round(p, 3)})
    fig.legend(handles=[Patch(facecolor='#34495e', label=f'Categoria Normal (>= {LIMIAR_RARIDADE_PCT}%)'),
                        Patch(facecolor='#e67e22', label=f'Categoria Rara (< {LIMIAR_RARIDADE_PCT}%)')],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.99), fontsize=11)
    plt.suptitle('Diagnóstico Visual de Categorias Raras (Rare Labels)', fontsize=15, y=1.02)
    fig.text(0.01, 0.005, f'Fonte: Inep, ENEM 2023 (desenvolvimento, n={total:,})', fontsize=8)
    fig.tight_layout()
    fig.savefig(destino_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tabela = pd.DataFrame(linhas_raras)
    tabela.to_csv(destino_tabela, index=False)
    print(f'Gráfico salvo: {destino_fig}')
    print(f'Categorias raras (< {LIMIAR_RARIDADE_PCT}%):\n{tabela.to_string(index=False)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    args = ap.parse_args()

    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.titlesize': 14})
    pasta_fig, pasta_tab = args.saida / 'figuras', args.saida / 'tabelas'
    pasta_fig.mkdir(parents=True, exist_ok=True)
    pasta_tab.mkdir(parents=True, exist_ok=True)

    df = carregar_desenvolvimento([ALVO, *FEATURES_PARA_ANALISE])
    print(f'Desenvolvimento carregado: {len(df):,} linhas (o teste não é lido).')
    plotar_distribuicao_alvo(df, pasta_fig / 'distribuicao_alvo.png')
    plotar_categorias_raras(df, pasta_fig / 'analise_categorias.png', pasta_tab / 'categorias_raras.csv')


if __name__ == '__main__':
    main()

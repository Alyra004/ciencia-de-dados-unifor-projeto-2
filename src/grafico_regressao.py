"""Gráfico previsão × observado de um modelo, com todos os participantes do desenvolvimento.

As previsões são "fora da dobra": `cross_val_predict` com o mesmo KFold(3, shuffle=True, random_state=42)
do protocolo, então cada participante é previsto por um modelo que não o viu no treino. Assim dá para
desenhar todos os pontos (sem amostra) sem olhar para o ajuste do treino. O arquivo de teste não é lido.

Uso (uma chamada por modelo, cada uma com seus parâmetros e suas features):
    avaliar_modelo('Ridge (alpha=1)', modelo, parametros, features, X, y, nomes_das_colunas)
"""
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from config import SEMENTE


def selecionar_colunas(X, nomes_todos, nomes_escolhidos):
    """Fatia as colunas de X pelos nomes (erro claro se algum nome não existir)."""
    posicao = {nome: i for i, nome in enumerate(nomes_todos)}
    faltando = [n for n in nomes_escolhidos if n not in posicao]
    if faltando:
        raise KeyError(f'Features que não existem na base: {faltando}')
    return X[:, [posicao[n] for n in nomes_escolhidos]]


def previsoes_fora_da_dobra(modelo, X, y):
    """Previsão de cada participante feita por um modelo ajustado só nas outras duas dobras."""
    cv = KFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    return cross_val_predict(clone(modelo), X, y, cv=cv, n_jobs=1)


def metricas(y, previsao):
    """RMSE, MAE e R² agregados nas três dobras, e a reta previsão = intercepto + inclinação × observado."""
    y, previsao = np.asarray(y, dtype=np.float64), np.asarray(previsao, dtype=np.float64)   # float32 estraga o ajuste da reta
    yc, pc = y - y.mean(), previsao - previsao.mean()
    inclinacao = float((yc * pc).sum() / (yc * yc).sum())
    intercepto = float(previsao.mean() - inclinacao * y.mean())
    return {'rmse': float(np.sqrt(mean_squared_error(y, previsao))), 'mae': float(mean_absolute_error(y, previsao)),
            'r2': float(r2_score(y, previsao)), 'inclinacao': float(inclinacao), 'intercepto': float(intercepto),
            'vies': float((y - previsao).mean())}


def _br(valor, casas, sinal=False):
    """Número com vírgula decimal (e sinal explícito se pedido)."""
    return f'{valor:{"+" if sinal else ""}.{casas}f}'.replace('.', ',')


def _milhar(n):
    return f'{int(n):,}'.replace(',', '.')


def _reta(m):
    """Texto da reta com o sinal certo ('+ 0,00' em vez de '+ -0,00' quando a inclinação é desprezível)."""
    inclinacao = 0.0 if round(m['inclinacao'], 2) == 0 else m['inclinacao']
    sinal = '−' if inclinacao < 0 else '+'
    return f'previsto = {m["intercepto"]:.0f} {sinal} {abs(inclinacao):.2f} × observado'.replace('.', ',')


def _slug(nome):
    ascii_ = unicodedata.normalize('NFKD', nome.lower()).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '_', ascii_).strip('_')


def _estimador_final(modelo):
    return modelo.steps[-1][1] if isinstance(modelo, Pipeline) else modelo


def descrever_features(features, n_total):
    if not features:
        return 'nenhuma (o modelo ignora X)'
    resumo = ', '.join(features[:4]) + (f' … (+{len(features) - 4})' if len(features) > 4 else '')
    return f'{len(features)} de {n_total} colunas: {resumo}'


def plotar_previsao_vs_observado(y, previsao, nome, parametros, descricao_features, m, preparacao=None, n_exemplos=40):
    """Esquerda: todos os pontos (hexágonos, cor = nº de participantes), linha ideal, reta de regressão e o erro de
    alguns participantes sorteados (traço vertical até a linha ideal). Direita: distribuição de todos os erros."""
    y, previsao = np.asarray(y, dtype=np.float64), np.asarray(previsao, dtype=np.float64)
    fig = plt.figure(figsize=(11.6, 6.2))
    grade = fig.add_gridspec(1, 2, width_ratios=[1.08, 1], left=0.07, right=0.985, top=0.79, bottom=0.12, wspace=0.3)
    ax, ax2 = fig.add_subplot(grade[0]), fig.add_subplot(grade[1])

    hb = ax.hexbin(y, previsao, gridsize=95, extent=(0, 1000, 0, 1000), mincnt=1, cmap='Blues', norm=LogNorm())
    x = np.array([0, 1000])
    ax.plot(x, x, color='#d62728', ls='--', lw=1.6, label='linha ideal (previsto = observado)')
    ax.plot(x, m['intercepto'] + m['inclinacao'] * x, color='#ff7f0e', lw=2.2, label=f'reta de regressão: {_reta(m)}')
    exemplos = np.random.RandomState(SEMENTE).choice(len(y), size=n_exemplos, replace=False)
    ax.vlines(y[exemplos], previsao[exemplos], y[exemplos], colors='#222222', lw=1.0, alpha=0.9, zorder=4)
    ax.scatter(y[exemplos], previsao[exemplos], s=16, color='#111111', zorder=5,
               label=f'{n_exemplos} participantes sorteados: o traço é o erro')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_box_aspect(1)
    ax.set_xlabel('Nota observada (NU_NOTA_MT)')
    ax.set_ylabel('Nota prevista (fora da dobra)')
    ax.set_title('Previsão × observado (todos os participantes)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=7.8, framealpha=0.95)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('participantes por hexágono (escala log)', fontsize=8.5)

    erro = y - previsao   # erro = observado − previsto = distância vertical do ponto até a linha ideal (com sinal)
    contagem, bordas = np.histogram(erro, bins=85, range=(-850, 850))
    assert contagem.sum() == len(erro), 'há erros fora do intervalo do histograma'
    centros = (bordas[:-1] + bordas[1:]) / 2
    ax2.bar(centros, contagem / 1e3, width=bordas[1] - bordas[0], color='#4C78A8', edgecolor='white', linewidth=0.3)
    for sinal in (-1, 1):
        ax2.axvline(sinal * m['mae'], color='#ff7f0e', ls='--', lw=1.8, label=f'± MAE = {_br(m["mae"], 1)}' if sinal > 0 else None)
        ax2.axvline(sinal * m['rmse'], color='#d62728', ls=':', lw=1.8, label=f'± RMSE = {_br(m["rmse"], 1)}' if sinal > 0 else None)
    ax2.set_xlim(-850, 850)
    ax2.set_ylim(0, contagem.max() / 1e3 * 1.75)   # folga em cima para o texto e a legenda não cobrirem as barras
    ax2.set_xlabel('Erro = observado − previsto (pontos)')
    ax2.set_ylabel('participantes (mil)')
    ax2.set_title('Distribuição dos erros (todos os participantes)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.25)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    r2 = 0.0 if round(m['r2'], 3) == 0 else m['r2']   # evita "-0,000"
    texto = (f'MAE = distância vertical média dos pontos\naté a linha ideal: {_br(m["mae"], 1)} pontos\n'
             f'RMSE: {_br(m["rmse"], 1)} pontos\nR²: {_br(r2, 3)}\nerro médio (viés): {_br(m["vies"], 1, sinal=True)}\n'
             f'n = {_milhar(len(y))}')
    ax2.text(0.03, 0.97, texto, transform=ax2.transAxes, ha='left', va='top', fontsize=8.3,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#999999', alpha=0.95))

    parametros_txt = ', '.join(f'{k}={v!r}' for k, v in parametros.items()) or '(nenhum)'
    cabecalho = [f'Parâmetros: {parametros_txt}', f'Features: {descricao_features}']
    if preparacao:
        cabecalho.append(f'Preparação: {preparacao}')
    cabecalho.append('Previsões fora da dobra: KFold(3, shuffle=True, random_state=42), só desenvolvimento')
    cabecalho = [textwrap.fill(linha, width=135, subsequent_indent='    ') for linha in cabecalho]
    fig.suptitle(nome, fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.07, 0.945, '\n'.join(cabecalho), ha='left', va='top', fontsize=8.5, color='#333333', linespacing=1.45)
    return fig


def avaliar_modelo(nome, modelo, parametros, features, X, y, nomes_todos, preparacao=None, pasta_figuras=None):
    """Imprime os parâmetros, calcula as previsões fora da dobra e desenha o gráfico.

    `features` é a lista de colunas usadas; vazia = o modelo não usa X (baseline), e X inteiro é passado só
    para manter as mesmas dobras. Devolve {'figura', 'previsao', 'metricas'}.
    """
    X_usado = selecionar_colunas(X, nomes_todos, features) if features else X
    descricao = descrever_features(features, len(nomes_todos))
    print(f'Modelo: {nome}')
    print(f'Parâmetros declarados: {parametros}')
    if preparacao:
        print(f'Preparação: {preparacao}')
    print(f'Features: {descricao}')
    print(f'Todos os parâmetros do estimador final ({type(_estimador_final(modelo)).__name__}): '
          f'{_estimador_final(modelo).get_params()}')
    print(f'Matriz usada: {X_usado.shape[0]:,} linhas × {X_usado.shape[1]} colunas')
    previsao = previsoes_fora_da_dobra(modelo, X_usado, y)
    m = metricas(y, previsao)
    fig = plotar_previsao_vs_observado(y, previsao, nome, parametros, descricao, m, preparacao)
    if pasta_figuras is not None:
        pasta = Path(pasta_figuras)
        pasta.mkdir(parents=True, exist_ok=True)
        fig.savefig(pasta / f'regressao_{_slug(nome)}.png', dpi=150)
    print(f'RMSE {m["rmse"]:.2f} | MAE {m["mae"]:.2f} | R² {m["r2"]:.4f} | reta: previsto = '
          f'{m["intercepto"]:.1f} + ({m["inclinacao"]:.3f}) × observado (agregado das 3 dobras)')
    return {'figura': fig, 'previsao': previsao, 'metricas': m}

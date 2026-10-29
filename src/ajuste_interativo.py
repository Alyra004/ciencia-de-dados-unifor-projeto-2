"""Painéis interativos: mexa nos parâmetros com o mouse e veja previsão e erro.

Treina em duas das três dobras do desenvolvimento e valida na terceira (a dobra 1 do mesmo
KFold(3, shuffle=True, random_state=42) do protocolo). O arquivo de teste nunca é carregado.
É uma exploração: a escolha oficial dos hiperparâmetros continua sendo feita pela validação
cruzada completa (GridSearchCV / RandomizedSearchCV).

No notebook 04 (modelo, hiperparâmetros e linhas de treino):
    from ajuste_interativo import painel, separar_dobra
    painel(separar_dobra(X_desenvolvimento, y_desenvolvimento))

No notebook 02 (features derivadas, seleção e alpha):
    from ajuste_interativo import painel_features, separar_dobra
    painel_features(separar_dobra(X_com_derivadas, y_dev), n_colunas_originais=X_dev.shape[1])

Pela linha de comando (figura estática, útil para os slides):
    python src/ajuste_interativo.py --alpha 10 --salvar resultados/figuras/ajuste_alpha10.png
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import SEMENTE
from modelagem_acompanhamento import carregar_desenvolvimento_esparso, criar_ridge

# Faixas de nota para analisar o erro (definidas só pelo desenvolvimento).
BORDAS_FAIXAS = [-1, 0.5, 400, 500, 600, 700, 800, 1001]
ROTULOS_FAIXAS = ['0 (zero)', '1–399', '400–499', '500–599', '600–699', '700–799', '800+']
MAX_PONTOS_DISPERSAO = 50_000
FAMILIAS = ['Ridge', 'Árvore de decisão', 'Gradient Boosting']
ULTIMOS_CONTROLES = {}   # controles do último painel criado (usado para testar os callbacks sem mouse)


@dataclass
class Dobra:
    X_tr: object
    y_tr: np.ndarray
    X_va: object
    y_va: np.ndarray
    idx_amostra: np.ndarray       # pontos usados só para desenhar a dispersão
    perm_treino: np.ndarray       # ordem aleatória fixa das linhas de treino (subamostras progressivas)
    rmse_baseline: float
    baseline_por_faixa: np.ndarray


def _rmse(erro):
    return float(np.sqrt(np.mean(np.square(erro, dtype=np.float64))))


def _rmse_por_faixa(y, previsao):
    """Devolve (RMSE por faixa, tamanho de cada faixa)."""
    faixa = np.digitize(y, BORDAS_FAIXAS[1:-1])
    rmse, n = np.full(len(ROTULOS_FAIXAS), np.nan), np.zeros(len(ROTULOS_FAIXAS), dtype=int)
    for i in range(len(ROTULOS_FAIXAS)):
        m = faixa == i
        n[i] = int(m.sum())
        if n[i]:
            rmse[i] = _rmse(y[m] - previsao[m])
    return rmse, n


def carregar_dobra(dobra=1, limite_linhas=None):
    """Carrega o desenvolvimento e separa treino e validação de uma dobra (padrão: a 1)."""
    X, y, _ = carregar_desenvolvimento_esparso(limite_linhas)
    return separar_dobra(X, y, dobra)


def separar_dobra(X, y, dobra=1):
    """Separa treino e validação de uma dobra a partir de X e y do desenvolvimento já carregados."""
    cv = KFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    treino, validacao = list(cv.split(X))[dobra - 1]
    X_tr, y_tr, X_va, y_va = X[treino], y[treino], X[validacao], y[validacao]
    media_treino = float(y_tr.mean())  # o baseline usa só a média do treino
    rng = np.random.RandomState(SEMENTE)
    idx = rng.choice(len(y_va), size=min(MAX_PONTOS_DISPERSAO, len(y_va)), replace=False)
    perm = rng.permutation(len(y_tr))
    base_faixa, _ = _rmse_por_faixa(y_va, np.full(len(y_va), media_treino))
    return Dobra(X_tr, y_tr, X_va, y_va, idx, perm, _rmse(y_va - media_treino), base_faixa)


# --- Modelos ---------------------------------------------------------------------
def _densificar(X):
    return X.toarray() if sparse.issparse(X) else X


def criar_ridge_com_selecao(alpha=1.0, k=None):
    """Ridge com SelectKBest opcional (k=None ou 0 = sem seleção); tudo dentro do Pipeline."""
    etapas = [('imputacao', SimpleImputer(strategy='median', keep_empty_features=True)),
              ('padronizacao', StandardScaler(with_mean=False))]
    if k:
        etapas.append(('selecao', SelectKBest(score_func=f_regression, k=k)))
    etapas.append(('modelo', Ridge(alpha=alpha, solver='lsqr')))
    return Pipeline(etapas)


def criar_modelo(familia, **p):
    """Devolve (estimador, rótulo) para a família e os hiperparâmetros dados."""
    if familia == 'Ridge':
        return criar_ridge(p['alpha']), f'Ridge(alpha={p["alpha"]:g})'
    if familia == 'Árvore de decisão':
        modelo = Pipeline([
            ('imputacao', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('modelo', DecisionTreeRegressor(max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'],
                                             random_state=SEMENTE))])
        return modelo, f'Árvore(max_depth={p["max_depth"]}, min_samples_leaf={p["min_samples_leaf"]})'
    if familia == 'Gradient Boosting':
        modelo = Pipeline([
            ('densificar', FunctionTransformer(_densificar, accept_sparse=True)),
            ('modelo', HistGradientBoostingRegressor(max_iter=p['max_iter'], learning_rate=p['learning_rate'],
                                                     max_depth=p['max_depth'], random_state=SEMENTE))])
        return modelo, (f'GradBoost(max_iter={p["max_iter"]}, lr={p["learning_rate"]:g}, '
                        f'max_depth={p["max_depth"]})')
    raise ValueError(f'Família desconhecida: {familia}')


def treinar_e_avaliar(modelo, dobra, n_treino=None, rotulo=None, familia=None):
    """Treina no treino da dobra (ou numa subamostra fixa dele) e mede na validação.

    `modelo` pode ser um número (alpha do Ridge, por compatibilidade) ou um estimador do sklearn.
    """
    alpha = None
    if isinstance(modelo, (int, float)):
        alpha = float(modelo)
        rotulo = rotulo or f'Ridge(alpha={alpha:g})'
        familia = familia or 'Ridge'
        modelo = criar_ridge(alpha)
    familia = familia or type(modelo).__name__
    rotulo = rotulo or familia

    if n_treino and n_treino < len(dobra.y_tr):
        linhas = np.sort(dobra.perm_treino[:n_treino])
        X_tr, y_tr = dobra.X_tr[linhas], dobra.y_tr[linhas]
    else:
        X_tr, y_tr = dobra.X_tr, dobra.y_tr

    inicio = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    tempo = time.perf_counter() - inicio
    previsao = np.asarray(modelo.predict(dobra.X_va), dtype=np.float64)
    erro = dobra.y_va - previsao
    rmse_faixa, n_faixa = _rmse_por_faixa(dobra.y_va, previsao)
    ss_res, ss_tot = float(np.sum(erro ** 2)), float(np.sum((dobra.y_va - dobra.y_va.mean()) ** 2))
    return {'alpha': alpha, 'rotulo': rotulo, 'familia': familia, 'n_treino': len(y_tr),
            'rmse': _rmse(erro), 'mae': float(np.mean(np.abs(erro))), 'r2': 1 - ss_res / ss_tot,
            'tempo': tempo, 'previsao': previsao, 'erro': erro,
            'rmse_faixa': rmse_faixa, 'n_faixa': n_faixa, 'modelo': modelo}


# --- Gráficos --------------------------------------------------------------------
def plotar(resultado, dobra, historico=None):
    """Quatro painéis: previsão × observado, resíduos, erro por faixa e RMSE de cada tentativa."""
    r = resultado
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    i = dobra.idx_amostra
    hb = ax[0, 0].hexbin(dobra.y_va[i], r['previsao'][i], gridsize=60, bins='log', mincnt=1, cmap='viridis')
    ax[0, 0].plot([0, 1000], [0, 1000], color='red', lw=1.5, ls='--', label='previsão perfeita')
    ax[0, 0].set(xlim=(0, 1000), ylim=(0, 1000), xlabel='Nota observada', ylabel='Nota prevista',
                 title=f'Previsão × observado (amostra de {len(i):,} da validação)')
    ax[0, 0].legend(loc='upper left')
    fig.colorbar(hb, ax=ax[0, 0], label='participantes (log)')

    ax[0, 1].hist(r['erro'], bins=80, color='#1f77b4', alpha=0.85)
    ax[0, 1].axvline(0, color='red', ls='--', lw=1.2)
    ax[0, 1].set(xlabel='Resíduo = observado − previsto (pontos)', ylabel='Participantes',
                 title=f'Resíduos (média {r["erro"].mean():+.1f}; desvio {r["erro"].std():.1f})')

    x = np.arange(len(ROTULOS_FAIXAS))
    ax[1, 0].bar(x - 0.2, dobra.baseline_por_faixa, width=0.4, color='#8c8c8c', label='baseline (média)')
    ax[1, 0].bar(x + 0.2, r['rmse_faixa'], width=0.4, color='#1f77b4', label=r['rotulo'][:38])
    for xi, n in zip(x, r['n_faixa']):
        ax[1, 0].annotate(f'n={n:,}', (xi, 0), xytext=(0, 4), textcoords='offset points',
                          ha='center', fontsize=7, color='white', fontweight='bold')
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(ROTULOS_FAIXAS, rotation=20)
    ax[1, 0].set(xlabel='Faixa da nota observada', ylabel='RMSE (pontos)', title='Erro por faixa de nota')
    ax[1, 0].legend()

    # Histórico: RMSE de cada tentativa, na ordem em que foram feitas (cor = família do modelo)
    historico = list(historico or [])
    cores = {f: plt.get_cmap('tab10')(k) for k, f in enumerate(FAMILIAS)}
    for k, h in enumerate(historico, start=1):
        ax[1, 1].scatter(k, h['rmse'], color=cores.get(h['familia'], '#555555'), s=45, zorder=3)
    ax[1, 1].scatter([len(historico)], [r['rmse']], s=160, facecolors='none', edgecolors='red', lw=2, zorder=5,
                     label='atual')
    ax[1, 1].axhline(dobra.rmse_baseline, color='#8c8c8c', ls='--', label='baseline')
    for f, c in cores.items():
        if any(h['familia'] == f for h in historico):
            ax[1, 1].scatter([], [], color=c, label=f)
    ax[1, 1].set(xlabel='Tentativa (ordem em que você treinou)', ylabel='RMSE na validação',
                 title='RMSE de cada tentativa')
    ax[1, 1].legend(fontsize=8)

    ganho = (1 - r['rmse'] / dobra.rmse_baseline) * 100
    fig.suptitle(f'{r["rotulo"]}  |  RMSE {r["rmse"]:.2f}  MAE {r["mae"]:.2f}  R² {r["r2"]:.3f}  '
                 f'|  baseline RMSE {dobra.rmse_baseline:.2f} ({ganho:.1f}% menor)  |  ajuste em {r["tempo"]:.1f}s',
                 fontsize=11, fontweight='bold')
    fig.text(0.01, 0.005, f'Fonte: Inep, ENEM 2023. Desenvolvimento: treino n={r["n_treino"]:,} '
                          f'(de {len(dobra.y_tr):,}), validação n={len(dobra.y_va):,} (dobra do KFold 3, '
                          f'semente {SEMENTE}). Teste não usado.', fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def plotar_previsoes(resultado, dobra, nomes_features, top=15):
    """Figura fixa da regressão: previsão × observado, resíduos × previsão e pesos do modelo.

    Painel 1: a linha tracejada é a previsão perfeita e a laranja é a nota prevista média em cada faixa
    de nota observada. Painel 3: pesos do Ridge nas features padronizadas (positivo = nota prevista
    maior) ou importâncias, para modelos de árvore.
    """
    r, i = resultado, dobra.idx_amostra
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

    hb = ax[0].hexbin(dobra.y_va[i], r['previsao'][i], gridsize=60, bins='log', mincnt=1, cmap='viridis')
    ax[0].plot([0, 1000], [0, 1000], color='red', lw=1.5, ls='--', label='previsão perfeita')
    bordas = np.arange(0, 1001, 50)
    centro, media = [], []
    for ini, fim in zip(bordas[1:-1], bordas[2:]):   # a faixa 0-50 (notas zero) vai à parte
        m = (dobra.y_va >= ini) & (dobra.y_va < fim)
        if m.sum() >= 200:
            centro.append((ini + fim) / 2)
            media.append(r['previsao'][m].mean())
    ax[0].plot(centro, media, color='orange', lw=2.2, marker='o', ms=4, label='previsão média por faixa observada')
    zeros = dobra.y_va == 0
    if zeros.any():
        ax[0].scatter([0], [r['previsao'][zeros].mean()], color='orange', s=70, marker='s', zorder=5,
                      edgecolor='black', label='previsão média das notas zero')
    ax[0].set(xlim=(0, 1000), ylim=(0, 1000), xlabel='Nota observada', ylabel='Nota prevista',
              title=f'Previsão × observado (amostra de {len(i):,})')
    ax[0].legend(loc='upper left', fontsize=8)
    fig.colorbar(hb, ax=ax[0], label='participantes (log)')

    hb2 = ax[1].hexbin(r['previsao'][i], r['erro'][i], gridsize=60, bins='log', mincnt=1, cmap='viridis')
    ax[1].axhline(0, color='red', ls='--', lw=1.3)
    ax[1].set(xlabel='Nota prevista', ylabel='Resíduo = observado − previsto',
              title=f'Resíduos × previsão (RMSE {r["rmse"]:.1f}; MAE {r["mae"]:.1f}; R² {r["r2"]:.3f})')
    fig.colorbar(hb2, ax=ax[1], label='participantes (log)')

    estimador = r['modelo'].steps[-1][1] if hasattr(r['modelo'], 'steps') else r['modelo']
    pesos = getattr(estimador, 'coef_', getattr(estimador, 'feature_importances_', None))
    if pesos is not None and np.size(pesos) == len(nomes_features):
        pesos = np.ravel(pesos)
        ordem = np.argsort(np.abs(pesos))[::-1][:top][::-1]
        cores = ['#2ca02c' if pesos[k] > 0 else '#d62728' for k in ordem]
        ax[2].barh([nomes_features[k] for k in ordem], pesos[ordem], color=cores)
        ax[2].axvline(0, color='black', lw=0.8)
        ax[2].set(xlabel='Peso (features padronizadas)', title=f'{top} maiores pesos: {r["rotulo"]}')
        ax[2].tick_params(axis='y', labelsize=8)
    else:
        ax[2].axis('off')
        ax[2].text(0.5, 0.5, 'Este modelo não expõe pesos\npor feature.', ha='center', va='center')

    fig.suptitle('Como o modelo prevê a nota de Matemática (validação, dobra 1 do desenvolvimento)',
                 fontsize=13, fontweight='bold')
    fig.text(0.01, 0.005, f'Fonte: Inep, ENEM 2023. Treino n={r["n_treino"]:,}; validação n={len(dobra.y_va):,}. '
                          'Teste não usado.', fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


# --- Painéis ---------------------------------------------------------------------
def _mostrar(saida, resultado, dobra, historico):
    from IPython.display import clear_output, display
    with saida:
        clear_output(wait=True)
        fig = plotar(resultado, dobra, historico)
        display(fig)
        plt.close(fig)


def painel(dobra, familia_inicial='Ridge'):
    """Escolha o modelo, ajuste os hiperparâmetros e as linhas de treino; ao soltar, treina e redesenha."""
    import ipywidgets as w
    from IPython.display import display

    estilo, largura = {'description_width': '120px'}, w.Layout(width='560px')
    n_total = len(dobra.y_tr)
    opcoes_n = sorted({n for n in (50_000, 100_000, 200_000, 500_000, 1_000_000) if n < n_total} | {n_total})

    familia = w.Dropdown(options=FAMILIAS, value=familia_inicial, description='Modelo', style=estilo)
    n_treino = w.SelectionSlider(options=opcoes_n, value=n_total, description='Linhas de treino',
                                 continuous_update=False, style=estilo, layout=largura)
    kw = dict(continuous_update=False, style=estilo, layout=largura)
    grupos = {
        'Ridge': {'alpha': w.FloatLogSlider(value=1.0, base=10, min=-3, max=8, step=0.25, description='alpha',
                                           readout_format='.3g', **kw)},
        'Árvore de decisão': {
            'max_depth': w.IntSlider(value=8, min=1, max=30, description='max_depth', **kw),
            'min_samples_leaf': w.SelectionSlider(options=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000], value=50,
                                                  description='min_samples_leaf', **kw)},
        'Gradient Boosting': {
            'max_iter': w.IntSlider(value=100, min=10, max=400, step=10, description='max_iter', **kw),
            'learning_rate': w.FloatLogSlider(value=0.1, base=10, min=-2, max=0, step=0.1, description='learning_rate',
                                              readout_format='.3f', **kw),
            'max_depth': w.IntSlider(value=6, min=2, max=12, description='max_depth', **kw)},
    }
    caixas = {f: w.VBox(list(s.values())) for f, s in grupos.items()}
    botao_limpar = w.Button(description='Limpar histórico', icon='trash')
    status, saida, historico = w.HTML(), w.Output(), []

    def exibir_grupo():
        for f, caixa in caixas.items():
            caixa.layout.display = None if f == familia.value else 'none'

    def atualizar(_=None):
        exibir_grupo()
        parametros = {nome: s.value for nome, s in grupos[familia.value].items()}
        estimador, rotulo = criar_modelo(familia.value, **parametros)
        status.value = f'<i>Treinando {rotulo} com {n_treino.value:,} linhas…</i>'
        resultado = treinar_e_avaliar(estimador, dobra, n_treino.value, rotulo=rotulo, familia=familia.value)
        historico.append({'rotulo': rotulo, 'rmse': resultado['rmse'], 'familia': familia.value})
        _mostrar(saida, resultado, dobra, historico)
        status.value = (f'<b>{rotulo}</b> &nbsp; {resultado["n_treino"]:,} linhas &nbsp; RMSE = <b>{resultado["rmse"]:.2f}</b>'
                        f' &nbsp; MAE = {resultado["mae"]:.2f} &nbsp; R² = {resultado["r2"]:.3f} '
                        f'&nbsp; ({resultado["tempo"]:.1f}s)')

    def trocar_modelo(_):
        # Árvore e boosting são mais lentos: começam com uma subamostra para a tela responder rápido.
        alvo = n_total if familia.value == 'Ridge' else min(200_000, n_total)
        n_treino.unobserve(atualizar, names='value')
        n_treino.value = min(opcoes_n, key=lambda n: abs(n - alvo))
        n_treino.observe(atualizar, names='value')
        atualizar()

    def limpar(_):
        historico.clear()
        atualizar()

    familia.observe(trocar_modelo, names='value')
    n_treino.observe(atualizar, names='value')
    for s in grupos.values():
        for controle in s.values():
            controle.observe(atualizar, names='value')
    botao_limpar.on_click(limpar)
    ULTIMOS_CONTROLES.update(familia=familia, n_treino=n_treino, grupos=grupos, status=status, historico=historico)
    exibir_grupo()
    display(w.VBox([w.HBox([familia, botao_limpar]), n_treino, *caixas.values(), status, saida]))
    atualizar()


def painel_features(dobra, n_colunas_originais):
    """Painel do notebook 02: liga/desliga as features derivadas, escolhe o k do SelectKBest e o alpha.

    `dobra` deve ser feita com a matriz que já tem as derivadas nas últimas colunas; sem elas, o painel
    usa só as primeiras `n_colunas_originais`.
    """
    import ipywidgets as w
    from IPython.display import display

    estilo, largura = {'description_width': '150px'}, w.Layout(width='560px')
    kw = dict(continuous_update=False, style=estilo, layout=largura)
    usar_derivadas = w.Checkbox(value=True, description='Incluir features derivadas', indent=False)
    k = w.IntSlider(value=0, min=0, max=dobra.X_tr.shape[1], step=5,
                    description='SelectKBest k (0 = sem)', **kw)
    alpha = w.FloatLogSlider(value=1.0, base=10, min=-3, max=8, step=0.25, description='alpha do Ridge',
                             readout_format='.3g', **kw)
    botao_limpar = w.Button(description='Limpar histórico', icon='trash')
    status, saida, historico = w.HTML(), w.Output(), []
    originais = (dobra.X_tr[:, :n_colunas_originais], dobra.X_va[:, :n_colunas_originais])

    def atualizar(_=None):
        derivadas = usar_derivadas.value
        k_efetivo = k.value if k.value > 0 else None
        k_max = dobra.X_tr.shape[1] if derivadas else n_colunas_originais
        if k_efetivo and k_efetivo > k_max:
            k_efetivo = k_max
        rotulo = (f'Ridge(alpha={alpha.value:g}) | {"com" if derivadas else "sem"} derivadas'
                  f' | {"k=" + str(k_efetivo) if k_efetivo else "sem seleção"}')
        status.value = f'<i>Treinando {rotulo}…</i>'
        alvo = dobra if derivadas else Dobra(originais[0], dobra.y_tr, originais[1], dobra.y_va, dobra.idx_amostra,
                                             dobra.perm_treino, dobra.rmse_baseline, dobra.baseline_por_faixa)
        resultado = treinar_e_avaliar(criar_ridge_com_selecao(alpha.value, k_efetivo), alvo, rotulo=rotulo,
                                      familia='Ridge')
        historico.append({'rotulo': rotulo, 'rmse': resultado['rmse'], 'familia': 'Ridge'})
        _mostrar(saida, resultado, dobra, historico)
        status.value = (f'<b>{rotulo}</b> &nbsp; RMSE = <b>{resultado["rmse"]:.3f}</b> &nbsp; '
                        f'MAE = {resultado["mae"]:.3f} &nbsp; R² = {resultado["r2"]:.4f} &nbsp; ({resultado["tempo"]:.1f}s)')

    def limpar(_):
        historico.clear()
        atualizar()

    for controle in (usar_derivadas, k, alpha):
        controle.observe(atualizar, names='value')
    botao_limpar.on_click(limpar)
    ULTIMOS_CONTROLES.update(usar_derivadas=usar_derivadas, k=k, alpha=alpha, status=status, historico=historico)
    display(w.VBox([usar_derivadas, k, alpha, botao_limpar, status, saida]))
    atualizar()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, nargs='+', default=[1.0], help='um ou mais valores de alpha')
    ap.add_argument('--salvar', type=Path, default=None, help='grava a figura do último alpha neste caminho')
    ap.add_argument('--limite-linhas', type=int, default=None, help='lê só as primeiras N linhas (teste rápido)')
    args = ap.parse_args()
    plt.switch_backend('Agg')

    dobra = carregar_dobra(limite_linhas=args.limite_linhas)
    print(f'Treino: {len(dobra.y_tr):,} | validação: {len(dobra.y_va):,} | RMSE baseline: {dobra.rmse_baseline:.2f}')
    historico = []
    for alpha in args.alpha:
        r = treinar_e_avaliar(alpha, dobra)
        historico.append({'rotulo': r['rotulo'], 'rmse': r['rmse'], 'familia': r['familia']})
        print(f'alpha={alpha:<8g} RMSE={r["rmse"]:.3f} MAE={r["mae"]:.3f} R²={r["r2"]:.4f} ({r["tempo"]:.1f}s)')
    if args.salvar:
        args.salvar.parent.mkdir(parents=True, exist_ok=True)
        plotar(r, dobra, historico).savefig(args.salvar, dpi=150)
        print(f'Figura salva em {args.salvar}')


if __name__ == '__main__':
    main()

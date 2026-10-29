"""Varredura exploratória do alpha do Ridge nos três conjuntos de features (A, B e D), com validação cruzada de 3 dobras.

Mesmas dobras e mesmo Pipeline do acompanhamento: KFold(3, shuffle=True, random_state=42) no DESENVOLVIMENTO, com
imputação pela mediana, fusão (só em D) e padronização esparsa reajustadas no treino de cada dobra. O teste não é lido.
Os conjuntos estão em src/conjuntos_features.py: A (138 originais), B (A + 5 derivadas) e D (B com os grupos fundidos).

Não é a busca oficial de hiperparâmetros (essa virá depois, com GridSearchCV/RandomizedSearchCV): serve para ver se o
alpha muda o erro em cada cenário. Cada alpha diferente de 1 vira uma linha em resultados/experimentos.csv (o alpha=1
de cada conjunto já está em acomp_ridge_alpha1/feat_A, feat_B e feat_D) e todos entram em
resultados/tabelas/ridge_alpha_cv.csv.

Uso:
    python src/varredura_alpha.py [--saida DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

from config import DIR_RESULTADOS, SEMENTE
from conjuntos_features import CONJUNTOS, colunas_finais, criar_pipeline, montar_conjunto
from modelagem_acompanhamento import DESCRICAO_FEATURES, SCORING, carregar_desenvolvimento_esparso
from registro_experimentos import linha_de_cv, registrar

ALPHAS = [0.001, 0.01, 1.0, 100.0, 10_000.0, 1_000_000.0, 10_000_000.0]
ALPHA_REFERENCIA = 1.0
PREPARACAO = {'A': 'imputação pela mediana + padronização esparsa dentro do Pipeline, por dobra',
              'B': 'imputação pela mediana + padronização esparsa dentro do Pipeline, por dobra',
              'D': 'imputação pela mediana + fusão por grupos + padronização esparsa dentro do Pipeline, por dobra'}
# alpha=1 de cada conjunto já registrado (usado só para conferir a reprodução)
REGISTRO_ALPHA1 = {'A': 'acomp_ridge_alpha1', 'B': 'feat_B_derivadas_ridge', 'D': 'feat_D_fusao_grupos_ridge'}


def id_do_alpha(conjunto, alpha):
    """Identificador estável. No conjunto A mantém o padrão antigo: 0.001 -> acomp_ridge_alpha0p001, 1e6 -> ...alpha1e06.
    Em B e D inclui o conjunto: acomp_ridge_B_alpha1e06."""
    texto = f'{alpha:g}'.replace('.', 'p').replace('+', '')
    return f'acomp_ridge_alpha{texto}' if conjunto == 'A' else f'acomp_ridge_{conjunto}_alpha{texto}'


def conclusao(resultado, referencia):
    """Texto objetivo: só compara com o alpha=1 do mesmo conjunto e descreve o que foi medido."""
    rmse, rmse_ref = -resultado['test_rmse'].mean(), -referencia['test_rmse'].mean()
    dif = rmse - rmse_ref
    if abs(dif) < 0.0005:
        return ('RMSE igual ao de alpha=1 (diferença menor que 0,001 ponto): com milhões de linhas e poucas colunas, '
                'a regularização quase não age.')
    texto = f'RMSE {abs(dif):.3f} ponto {"acima" if dif > 0 else "abaixo"} do de alpha=1'.replace('.', ',')
    if dif > 0.5:
        r2 = f'{resultado["test_r2"].mean():.3f}'.replace('.', ',')
        texto += f'; o modelo fica rígido demais (R² cai para {r2}), ou seja, subajusta.'
    else:
        texto += '; diferença pequena frente ao desvio entre dobras de cerca de 0,17.'
    return texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    args = ap.parse_args()

    X, y, nomes = carregar_desenvolvimento_esparso()
    cv = KFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    registrados = pd.read_csv(args.saida / 'experimentos.csv').set_index('id_experimento')
    linhas, tabela = [], []
    for conjunto, descricao in CONJUNTOS.items():
        X_conj, nomes_matriz = montar_conjunto(conjunto, X, nomes)
        n_colunas = len(colunas_finais(conjunto, nomes_matriz))
        resultados = {}
        for alpha in ALPHAS:
            print(f'Validação cruzada: Ridge(alpha={alpha:g}) no conjunto {conjunto}', flush=True)
            pipeline = criar_pipeline(conjunto, Ridge(alpha=alpha, solver='lsqr'), nomes_matriz)
            resultados[alpha] = cross_validate(pipeline, X_conj, y, cv=cv, scoring=SCORING, n_jobs=1, error_score='raise')
        referencia = resultados[ALPHA_REFERENCIA]
        antigo = registrados.loc[REGISTRO_ALPHA1[conjunto], 'rmse_media']
        assert abs(-referencia['test_rmse'].mean() - antigo) < 1e-3, f'conjunto {conjunto}: alpha=1 não reproduz {REGISTRO_ALPHA1[conjunto]}'
        for alpha, r in resultados.items():
            rmse, mae, r2 = -r['test_rmse'], -r['test_mae'], r['test_r2']
            tabela.append({'conjunto': conjunto, 'n_colunas': n_colunas, 'alpha': alpha, 'rmse_media': rmse.mean(),
                           'rmse_dp': rmse.std(ddof=1), 'mae_media': mae.mean(), 'mae_dp': mae.std(ddof=1),
                           'r2_media': r2.mean(), 'r2_dp': r2.std(ddof=1), 'tempo_ajuste_total_s': float(np.sum(r['fit_time']))})
            if alpha != ALPHA_REFERENCIA:
                features = DESCRICAO_FEATURES if conjunto == 'A' else f'conjunto {conjunto}: {descricao}'
                linhas.append(linha_de_cv(
                    id_do_alpha(conjunto, alpha),
                    f'Um alpha diferente de 1 (alpha={alpha:g}) muda o erro do Ridge no conjunto {conjunto}',
                    features, PREPARACAO[conjunto], 'Ridge', f'alpha={alpha:g}; solver=lsqr', r, conclusao(r, referencia)))
    registrar(linhas, pasta_saida=args.saida)
    pasta = args.saida / 'tabelas'
    pasta.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(tabela)
    df.to_csv(pasta / 'ridge_alpha_cv.csv', index=False)
    print('\n' + df[['conjunto', 'n_colunas', 'alpha', 'rmse_media', 'rmse_dp', 'mae_media', 'r2_media']].to_string(index=False))
    print(f'\n{len(linhas)} experimentos registrados em {args.saida / "experimentos.csv"}; tabela em {pasta / "ridge_alpha_cv.csv"}')


if __name__ == '__main__':
    main()

"""Cada modelo treinado em cada conjunto de features (A, B e D), com as mesmas dobras e o mesmo protocolo.

Validação cruzada de 3 dobras (KFold(3, shuffle=True, random_state=42)) no DESENVOLVIMENTO; o teste não é lido.
Modelos treinados até agora: baseline pela média e Ridge. Para incluir outro modelo, acrescente uma entrada em MODELOS.

O Ridge nos conjuntos A, B e D já estava em resultados/experimentos.csv (feat_A, feat_B e feat_D); aqui ele é
refeito pelo mesmo caminho dos demais modelos e conferido contra esses valores. O baseline nos conjuntos B e D é novo
e vira linha nova em experimentos.csv. A tabela completa vai para resultados/tabelas/modelos_por_conjunto.csv.

Uso:
    python src/modelos_por_conjunto.py [--saida DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

from config import DIR_RESULTADOS, SEMENTE
from conjuntos_features import CONJUNTOS, colunas_finais, criar_pipeline, montar_conjunto
from modelagem_acompanhamento import SCORING, carregar_desenvolvimento_esparso
from registro_experimentos import linha_de_cv, registrar

MODELOS = {
    'Baseline (média)': {'criar': lambda: DummyRegressor(strategy='mean'), 'parametros': "strategy='mean'", 'preparar': False},
    'Ridge': {'criar': lambda: Ridge(alpha=1.0, solver='lsqr'), 'parametros': "alpha=1.0; solver='lsqr'", 'preparar': True},
}
# Linhas que já existem em experimentos.csv (usadas só para conferir a reprodução).
JA_REGISTRADOS = {('Baseline (média)', 'A'): 'acomp_baseline_media', ('Ridge', 'A'): 'feat_A_originais_ridge',
                  ('Ridge', 'B'): 'feat_B_derivadas_ridge', ('Ridge', 'D'): 'feat_D_fusao_grupos_ridge'}
ID_NOVO = {('Baseline (média)', 'B'): 'grade_baseline_media_B', ('Baseline (média)', 'D'): 'grade_baseline_media_D'}
PREPARACAO = {True: 'imputação pela mediana (+ fusão em D) + padronização esparsa dentro do Pipeline, por dobra',
              False: 'nenhuma (o baseline ignora as features)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_RESULTADOS, help='raiz de saída (padrão: resultados/)')
    args = ap.parse_args()

    X, y, nomes = carregar_desenvolvimento_esparso()
    cv = KFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    registrados = pd.read_csv(args.saida / 'experimentos.csv').set_index('id_experimento')
    resultados, linhas_tabela, novas = {}, [], []
    for conjunto, descricao in CONJUNTOS.items():
        X_conj, nomes_matriz = montar_conjunto(conjunto, X, nomes)
        n_colunas = len(colunas_finais(conjunto, nomes_matriz))
        for modelo, cfg in MODELOS.items():
            print(f'Validação cruzada: {modelo} no conjunto {conjunto} ({n_colunas} colunas)', flush=True)
            pipeline = criar_pipeline(conjunto, cfg['criar'](), nomes_matriz, preparar=cfg['preparar'])
            r = cross_validate(pipeline, X_conj, y, cv=cv, scoring=SCORING, n_jobs=1, error_score='raise')
            resultados[(modelo, conjunto)] = r
            rmse, mae, r2 = -r['test_rmse'], -r['test_mae'], r['test_r2']
            chave = (modelo, conjunto)
            linhas_tabela.append({'modelo': modelo, 'conjunto': conjunto, 'descricao_conjunto': descricao, 'n_colunas': n_colunas,
                                  'parametros': cfg['parametros'], 'rmse_media': rmse.mean(), 'rmse_dp': rmse.std(ddof=1),
                                  'mae_media': mae.mean(), 'mae_dp': mae.std(ddof=1), 'r2_media': r2.mean(), 'r2_dp': r2.std(ddof=1),
                                  'id_experimento': JA_REGISTRADOS.get(chave) or ID_NOVO.get(chave)})
            if chave in JA_REGISTRADOS:
                antigo = registrados.loc[JA_REGISTRADOS[chave], 'rmse_media']
                print(f'   confere com {JA_REGISTRADOS[chave]}: RMSE {rmse.mean():.4f} contra {antigo:.4f} '
                      f'({"ok" if abs(rmse.mean() - antigo) < 1e-3 else "DIFERENTE"})', flush=True)
                assert abs(rmse.mean() - antigo) < 1e-3, f'{chave} não reproduz o valor já registrado'

    ref = resultados[('Baseline (média)', 'A')]
    for (modelo, conjunto), id_ in ID_NOVO.items():
        r = resultados[(modelo, conjunto)]
        igual = np.allclose(-r['test_rmse'], -ref['test_rmse'], atol=1e-9)
        novas.append(linha_de_cv(
            id_, f'O baseline com o conjunto {conjunto} dá o mesmo erro que com o conjunto A, porque ignora as features',
            f'conjunto {conjunto}: {CONJUNTOS[conjunto]}; nenhuma feature entra na previsão', PREPARACAO[False], 'DummyRegressor',
            "strategy='mean'", r,
            'RMSE, MAE e R² iguais aos do baseline com o conjunto A, nas três dobras: o baseline ignora as features.' if igual
            else 'Atenção: o resultado difere do baseline com o conjunto A; verificar.'))
    registrar(novas, pasta_saida=args.saida)

    pasta = args.saida / 'tabelas'
    pasta.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(linhas_tabela)
    df.to_csv(pasta / 'modelos_por_conjunto.csv', index=False)
    print('\n' + df[['modelo', 'conjunto', 'n_colunas', 'rmse_media', 'rmse_dp', 'mae_media', 'r2_media']].to_string(index=False))
    print(f'\n{len(novas)} experimentos novos registrados; tabela em {pasta / "modelos_por_conjunto.csv"}')


if __name__ == '__main__':
    main()

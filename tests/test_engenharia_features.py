"""Testes das features: rodar com `python tests/test_engenharia_features.py` ou pytest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from engenharia_features import FEATURES_CRIADAS, EngenhariaFeatures, criar_features


def _linha(**kw):
    base = {'Q001': 'E', 'Q002': 'G', 'Q005': 4, 'Q006': 'H', 'Q007': 'A', 'Q008': 'B', 'Q009': 'C',
            'Q022': 'C', 'Q024': 'B', 'Q025': 'B', 'TP_ESCOLA': 2, 'TP_FAIXA_ETARIA': 3}
    base.update({f'Q{i:03d}': 'B' for i in range(10, 22)})
    base['Q023'] = 'A'
    base.update(kw)
    return pd.DataFrame([base])


def test_valores_esperados():
    o = criar_features(_linha()).iloc[0]
    assert o.ESC_MAX_RESP == 6 and o.ESC_DIFERENCA == 2
    assert o.RENDA_SM == 4.5 and o.RENDA_PER_CAPITA_SM == 4.5 / 4
    assert o.INDICE_BENS == 12 and o.MORADORES_POR_QUARTO == 2
    assert o.INDICE_DIGITAL == 3 and o.ESCOLA_PUBLICA == 1 and o.IDADE_FORA_FAIXA_TIPICA == 0
    assert 0 <= o.INDICE_SES <= 1


def test_desconhecimento_e_denominador_invalido():
    o = criar_features(_linha(Q001='H', Q009='A', Q005=0)).iloc[0]
    assert np.isnan(o.ESC_PAI) and o.ESC_MAX_RESP == 6 and o.ESC_DESCONHECE_ALGUM == 1
    assert np.isnan(o.MORADORES_POR_QUARTO) and np.isnan(o.RENDA_PER_CAPITA_SM)


def test_codigos_como_texto():
    # O CSV lido com dtype=str não pode zerar ESCOLA_PUBLICA nem IDADE_FORA_FAIXA_TIPICA.
    o = criar_features(_linha(TP_ESCOLA='2', TP_FAIXA_ETARIA='5', Q005='4')).iloc[0]
    assert o.ESCOLA_PUBLICA == 1 and o.IDADE_FORA_FAIXA_TIPICA == 1 and o.RENDA_PER_CAPITA_SM == 4.5 / 4


def test_sem_estado_e_colunas():
    df = _linha()
    out = EngenhariaFeatures().fit(df).transform(df)
    assert set(FEATURES_CRIADAS) <= set(out.columns) and len(df.columns) < len(out.columns)


if __name__ == '__main__':
    for nome, f in list(globals().items()):
        if nome.startswith('test_'):
            f()
            print('ok', nome)

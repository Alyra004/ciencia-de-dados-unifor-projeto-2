"""Engenharia de features para a previsão de NU_NOTA_MT (ENEM 2023).

Todas as features aqui são determinísticas e linha a linha: não aprendem nada
com os dados (sem médias, sem codificação supervisionada). Por isso não geram
vazamento e podem ser aplicadas antes ou dentro do Pipeline. Transformações que
aprendem com os dados (padronização, PCA, seleção) ficam no Pipeline de
modelagem, ajustadas apenas no treino de cada dobra.

Escalas dos códigos conferidas contra o dicionário do ENEM 2023. Ausência
(NaN/Nao_Informado) e desconhecimento ('H' em Q001/Q002) viram NaN nas
numéricas e são sinalizados à parte; o imputador fica no Pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Q001/Q002: escolaridade dos responsáveis -------------------------------
# A nunca estudou ... G pós-graduação; H = "não sei" (desconhecimento, não é nível)
ESCOLARIDADE = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
ESC_DESCONHECE = 'H'

# --- Q006: faixa de renda familiar, em salários mínimos ---------------------
# Aproximação pelo ponto médio da faixa. A = nenhuma renda. Q (> 20 SM) usa o
# piso (20): é um limite inferior, o que subestima as rendas mais altas.
RENDA_SM = {
    'A': 0.0, 'B': 0.5, 'C': 1.25, 'D': 1.75, 'E': 2.25, 'F': 2.75, 'G': 3.5,
    'H': 4.5, 'I': 5.5, 'J': 6.5, 'K': 7.5, 'L': 8.5, 'M': 9.5, 'N': 11.0,
    'O': 13.5, 'P': 17.5, 'Q': 20.0,
}

# Itens de contagem do domicílio: A = nenhum, B = 1, C = 2, D = 3, E = 4 ou mais.
# O último nível é "N ou mais", então a contagem é um piso. Q018, Q020, Q021 e
# Q023 são binários (A = não, B = sim) e usam o mesmo mapa.
CONTAGEM = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
# Q010 carro, Q011 moto, Q012 geladeira, Q013 freezer, Q014 lavar roupa,
# Q015 secar roupa, Q016 micro-ondas, Q017 lava-louça, Q018 aspirador,
# Q019 TV, Q020 DVD, Q021 TV por assinatura, Q023 telefone fixo.
# Celular (Q022), computador (Q024) e internet (Q025) ficam no índice digital.
BENS_DURAVEIS = ['Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018',
                 'Q019', 'Q020', 'Q021', 'Q023']
BENS_MAX = 9 * 4 + 4 * 1  # teto da soma: 9 itens com até 4 e 4 itens binários

Q_MORADORES = 'Q005'   # número de moradores (1 a 20)
Q_EMPREGADA = 'Q007'   # A = não, B/C/D = dias por semana
Q_BANHEIROS = 'Q008'   # A = nenhum ... E = 4 ou mais
Q_QUARTOS = 'Q009'     # quartos para dormir, mesma escala de contagem
Q_CARRO, Q_MOTO = 'Q010', 'Q011'
Q_CELULAR = 'Q022'     # celulares
Q_COMPUTADOR = 'Q024'  # computadores
Q_INTERNET = 'Q025'    # A = não, B = sim

NAO_INFORMADO = 'Nao_Informado'

FEATURES_CRIADAS = [
    'ESC_PAI', 'ESC_MAE', 'ESC_MAX_RESP', 'ESC_MEDIA_RESP', 'ESC_DIFERENCA',
    'ESC_DESCONHECE_ALGUM',
    'RENDA_SM', 'RENDA_PER_CAPITA_SM', 'RENDA_PER_CAPITA_LOG',
    'INDICE_BENS', 'BENS_PER_CAPITA', 'VEICULOS', 'BANHEIROS', 'EMPREGADA_DOMESTICA',
    'INDICE_DIGITAL', 'COMPUTADOR_PER_CAPITA',
    'MORADORES_POR_QUARTO',
    'RENDA_X_ESC_MAX', 'INDICE_SES',
    'ESCOLA_PUBLICA', 'IDADE_FORA_FAIXA_TIPICA',
]


def _num(serie):
    """Converte para número (o CSV pode trazer códigos como texto); inválido vira NaN."""
    return pd.to_numeric(serie, errors='coerce').astype('float32')


def _mapear(serie, tabela):
    """Mapeia letras para números; qualquer outra coisa (inclui Nao_Informado) vira NaN."""
    return serie.astype(str).map(tabela).astype('float32')


def _dividir(num, den):
    """Divisão com denominador inválido (<= 0 ou ausente) explicitado como NaN."""
    den = den.where(den > 0)
    return (num / den).astype('float32')


def criar_features(df):
    """Devolve uma cópia de df com as FEATURES_CRIADAS acrescentadas."""
    out = df.copy()
    q = lambda col: out[col].astype(str)  # noqa: E731  (Q* pode vir como category)

    # 1) Escolaridade dos responsáveis. 'H' (não sei) fica NaN e é sinalizado à parte,
    #    separado da ausência (Nao_Informado vira NaN em _mapear), como pede o Apêndice A.
    out['ESC_PAI'] = _mapear(out['Q001'], ESCOLARIDADE)
    out['ESC_MAE'] = _mapear(out['Q002'], ESCOLARIDADE)
    out['ESC_MAX_RESP'] = out[['ESC_PAI', 'ESC_MAE']].max(axis=1).astype('float32')
    out['ESC_MEDIA_RESP'] = out[['ESC_PAI', 'ESC_MAE']].mean(axis=1).astype('float32')
    out['ESC_DIFERENCA'] = (out['ESC_PAI'] - out['ESC_MAE']).abs().astype('float32')
    out['ESC_DESCONHECE_ALGUM'] = ((q('Q001') == ESC_DESCONHECE) | (q('Q002') == ESC_DESCONHECE)).astype('int8')

    # 2) Renda familiar e renda per capita. Denominador = moradores (>= 1).
    moradores = _num(out[Q_MORADORES])
    out['RENDA_SM'] = _mapear(out['Q006'], RENDA_SM)
    out['RENDA_PER_CAPITA_SM'] = _dividir(out['RENDA_SM'], moradores)
    out['RENDA_PER_CAPITA_LOG'] = np.log1p(out['RENDA_PER_CAPITA_SM']).astype('float32')

    # 3) Bens duráveis: soma (piso) e por morador; veículos, banheiros e empregada à parte.
    bens = pd.concat([_mapear(out[c], CONTAGEM) for c in BENS_DURAVEIS], axis=1)
    out['INDICE_BENS'] = bens.sum(axis=1, min_count=1).astype('float32')
    out['BENS_PER_CAPITA'] = _dividir(out['INDICE_BENS'], moradores)
    out['VEICULOS'] = (_mapear(out[Q_CARRO], CONTAGEM) + _mapear(out[Q_MOTO], CONTAGEM)).astype('float32')
    out['BANHEIROS'] = _mapear(out[Q_BANHEIROS], CONTAGEM)
    # Q007: qualquer dia de empregada (B, C, D) = 1; A = 0; ausente = NaN.
    out['EMPREGADA_DOMESTICA'] = (q(Q_EMPREGADA).map({'A': 0, 'B': 1, 'C': 1, 'D': 1})).astype('float32')

    # 4) Acesso digital: celular + computador + internet (0 a 3) e computador por morador.
    tem_celular = (_mapear(out[Q_CELULAR], CONTAGEM) > 0)
    comp = _mapear(out[Q_COMPUTADOR], CONTAGEM)
    tem_comp = comp > 0
    tem_net = q(Q_INTERNET) == 'B'
    out['INDICE_DIGITAL'] = (tem_celular.astype(int) + tem_comp.astype(int) + tem_net.astype(int)).astype('int8')
    out['COMPUTADOR_PER_CAPITA'] = _dividir(comp, moradores)

    # 5) Adensamento: moradores por quarto (quartos = 0 ou ausente => NaN).
    quartos = _mapear(out[Q_QUARTOS], CONTAGEM)
    out['MORADORES_POR_QUARTO'] = _dividir(moradores, quartos)

    # 6) Interação renda x escolaridade (hipótese: efeito combinado de capital econômico e cultural).
    out['RENDA_X_ESC_MAX'] = (out['RENDA_SM'] * out['ESC_MAX_RESP']).astype('float32')

    # 7) Índice socioeconômico composto: média das dimensões em [0, 1], com constantes
    #    fixas (nada aprendido dos dados). Dimensão ausente é ignorada na média.
    dimensoes = pd.concat([
        out['ESC_MAX_RESP'] / max(ESCOLARIDADE.values()),
        out['RENDA_PER_CAPITA_LOG'] / np.log1p(max(RENDA_SM.values())),
        out['INDICE_BENS'] / BENS_MAX,
        out['INDICE_DIGITAL'] / 3,
    ], axis=1).clip(0, 1)
    out['INDICE_SES'] = dimensoes.mean(axis=1).astype('float32')

    # 8) Contexto escolar. TP_ESCOLA: 1 = não respondeu, 2 = pública, 3 = privada.
    out['ESCOLA_PUBLICA'] = (_num(out['TP_ESCOLA']) == 2).astype('int8')
    # TP_FAIXA_ETARIA: 1 = < 17, 2 = 17, 3 = 18, 4 = 19 ... (acima de 18 sugere defasagem/EJA)
    out['IDADE_FORA_FAIXA_TIPICA'] = (_num(out['TP_FAIXA_ETARIA']) > 3).astype('int8')

    return out


class EngenhariaFeatures(BaseEstimator, TransformerMixin):
    """Transformer sem estado para uso dentro de sklearn.pipeline.Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return criar_features(X)

"""Fusão dos grupos de features redundantes sugeridos no notebook 02.

Os grupos vêm do FeatureAgglomeration aplicado às colunas ordinais/contínuas e derivadas
(limiar |rho de Spearman| >= 0,85, amostra de 100 mil do desenvolvimento). Cada grupo vira
uma única coluna: a média das colunas do grupo padronizadas (z-score). A média e o desvio de
cada coluna são aprendidos só no treino, dentro do Pipeline, então não há vazamento entre dobras.

Uso (o transformador vem depois da imputação):
    grupos = indices_dos_grupos(nomes_das_colunas)
    Pipeline([('imputacao', SimpleImputer(...)), ('fusao', FusaoPorGrupos(grupos)), ...])
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# Resultado do FeatureAgglomeration; o par Q024 x INDICE_DIGITAL passa de 0,85 mas não formou grupo.
GRUPOS_FUSAO = {
    'FUSAO_RENDA': ['Q006_RENDA_SM_ESTIMADA', 'RENDA_PER_CAPITA', 'RENDA_X_ESC_MAX'],
    'FUSAO_ESCOLARIDADE': ['Q002_NIVEL_ESCOLARIDADE', 'ESC_MAX_RESP'],
}


def indices_dos_grupos(nomes, grupos=GRUPOS_FUSAO):
    """Posição de cada coluna de cada grupo em `nomes` (tupla de tuplas, como o transformador espera)."""
    posicao = {nome: i for i, nome in enumerate(nomes)}
    ausentes = [c for membros in grupos.values() for c in membros if c not in posicao]
    if ausentes:
        raise KeyError(f'Colunas do grupo ausentes em X: {ausentes}')
    return tuple(tuple(posicao[c] for c in membros) for membros in grupos.values())


def nomes_apos_fusao(nomes, grupos=GRUPOS_FUSAO):
    """Nomes das colunas depois da fusão: as não fundidas, na ordem original, e depois uma por grupo."""
    fundidas = {c for membros in grupos.values() for c in membros}
    return [n for n in nomes if n not in fundidas] + list(grupos)


def _como_denso(bloco):
    return (bloco.toarray() if sparse.issparse(bloco) else np.asarray(bloco)).astype(np.float64)


class FusaoPorGrupos(BaseEstimator, TransformerMixin):
    """Troca as colunas de cada grupo pela média delas padronizadas; as demais passam sem mudança."""

    def __init__(self, grupos):
        self.grupos = grupos

    def fit(self, X, y=None):
        self.estatisticas_ = []
        for indices in self.grupos:
            bloco = _como_denso(X[:, list(indices)])
            desvio = bloco.std(axis=0)
            self.estatisticas_.append((bloco.mean(axis=0), np.where(desvio > 0, desvio, 1.0)))
        usadas = {i for indices in self.grupos for i in indices}
        self.indices_mantidos_ = [i for i in range(X.shape[1]) if i not in usadas]
        return self

    def transform(self, X):
        fundidas = [((_como_denso(X[:, list(indices)]) - media) / desvio).mean(axis=1, keepdims=True)
                    for indices, (media, desvio) in zip(self.grupos, self.estatisticas_)]
        fundidas = np.hstack(fundidas).astype(np.float32)
        resto = X[:, self.indices_mantidos_]
        if sparse.issparse(X):
            return sparse.hstack([resto, sparse.csr_matrix(fundidas)], format='csr', dtype=np.float32)
        return np.hstack([resto, fundidas])

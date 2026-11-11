"""Classes e funções da solução final, para carregar os modelos salvos em models/ fora do notebook.

Os pipelines salvos com joblib guardam a classe FusaoSituacao pelo nome `__main__.FusaoSituacao` (ela foi definida
no notebook 04). Para carregar um modelo em outro script, registre a classe antes:

    import __main__, joblib
    from src.modelagem import FusaoSituacao
    __main__.FusaoSituacao = FusaoSituacao
    modelo = joblib.load("models/final_avaliacao_final.joblib")
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

COLS_SITUACAO = ["IN_TREINEIRO", "TP_ANO_CONCLUIU", "TP_FAIXA_ETARIA", "TP_ST_CONCLUSAO"]


class FusaoSituacao(BaseEstimator, TransformerMixin):
    """Troca as 4 colunas de situação escolar por FUSAO_1, com média e desvio aprendidos só no treino."""
    colunas = COLS_SITUACAO
    sinais = np.array([1, -1, -1, 1])           # IN_TREINEIRO, TP_ANO_CONCLUIU, TP_FAIXA_ETARIA, TP_ST_CONCLUSAO

    def _z(self, X):
        z = X[self.colunas].astype("float64")
        z["TP_ANO_CONCLUIU"] = z["TP_ANO_CONCLUIU"].mask(z["TP_ANO_CONCLUIU"] == 0)     # 0 = "não informado" vira ausente
        return z

    def fit(self, X, y=None):
        z = self._z(X)
        self.media_, self.desvio_ = z.mean(), z.std()
        return self

    def transform(self, X):
        if not set(self.colunas) <= set(X.columns):
            return X
        fusao = (((self._z(X) - self.media_) / self.desvio_) * self.sinais).mean(axis=1)
        saida = X.drop(columns=self.colunas)
        saida["FUSAO_1"] = fusao.to_numpy()
        return saida

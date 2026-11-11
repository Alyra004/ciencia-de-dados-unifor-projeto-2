"""Funções de apoio dos notebooks 06 e 07: protocolo (partição e dobras), validação cruzada com métricas por dobra e o registro em
resultados/experimentos.csv. Os caminhos são relativos à pasta notebooks/ (de onde os notebooks rodam)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit

PROC = Path("../dados/processed")
COLUNAS_EXP = ["secao", "hipotese", "features", "preparacao", "modelo", "parametros", "avaliado", "n_features",
               "RMSE_d1", "RMSE_d2", "RMSE_d3", "MAE_d1", "MAE_d2", "MAE_d3", "R2_d1", "R2_d2", "R2_d3",
               "RMSE_media", "RMSE_desvio", "MAE_media", "MAE_desvio", "R2_media", "R2_desvio", "tempo_s", "conclusao"]


def protocolo():
    """ids dos elegíveis, máscara do desenvolvimento e as 3 dobras fixas (dados/splits/)."""
    ids = pd.read_csv(PROC / "ids_elegiveis.csv")["NU_INSCRICAO"]
    em_dev = ids.isin(pd.read_csv("../dados/splits/desenvolvimento_ids.csv")["NU_INSCRICAO"]).to_numpy()
    dobras = PredefinedSplit(pd.read_csv("../dados/splits/dobras_cv.csv")["dobra"].to_numpy())
    return ids, em_dev, dobras


def metricas(real, prev):
    return {"RMSE": mean_squared_error(real, prev) ** 0.5, "MAE": mean_absolute_error(real, prev), "R2": r2_score(real, prev)}


def cv_dobras(modelo, X, y, dobras):
    """Treina em 2 dobras e valida na 3ª, três vezes; devolve uma linha por dobra (RMSE, MAE, R2, tempo_s)."""
    linhas = []
    for tr, va in dobras.split():
        t0 = time.time()
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        r = metricas(y.iloc[va], m.predict(X.iloc[va]))
        r["tempo_s"] = time.time() - t0
        linhas.append(r)
    return pd.DataFrame(linhas)


class Registro:
    """Lê e grava resultados/experimentos.csv (uma linha por experimento, métricas por dobra incluídas)."""

    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self.exp = pd.read_csv(self.caminho, index_col="id") if self.caminho.exists() else pd.DataFrame(columns=COLUNAS_EXP)

    def tem_dobras(self, id_exp):
        return id_exp in self.exp.index and not pd.isna(self.exp.loc[id_exp, "RMSE_d1"])

    def gravar(self, id_exp, d=None, **campos):
        """Cria ou atualiza a linha `id_exp`. `d` (saída de cv_dobras) preenche as métricas por dobra, a média e o desvio."""
        linha = self.exp.loc[id_exp].to_dict() if id_exp in self.exp.index else {c: np.nan for c in COLUNAS_EXP}
        linha.update(campos)
        if d is not None:
            for m in ("RMSE", "MAE", "R2"):
                for k in range(3):
                    linha[f"{m}_d{k + 1}"] = d[m].iloc[k]
                linha[f"{m}_media"], linha[f"{m}_desvio"] = d[m].mean(), d[m].std()
            linha["tempo_s"] = d["tempo_s"].mean()
        self.exp.loc[id_exp] = pd.Series(linha).reindex(COLUNAS_EXP)
        self.exp.to_csv(self.caminho, index_label="id")
        return self.exp.loc[id_exp]

#!/usr/bin/env python3
"""Treino reproduzível dos regressores do Projeto 2 usando a base ENEM completa."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "dados/raw/DADOS/MICRODADOS_ENEM_2023.csv"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
TARGET = "NU_NOTA_MT"
ID = "NU_INSCRICAO"

# Apenas atributos de perfil/contexto e questionário. Colunas de desempenho,
# presença, identificação do caderno, respostas e gabaritos não entram em X.
CAT_FEATURES = [
    "TP_FAIXA_ETARIA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA",
    "TP_NACIONALIDADE", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_ESCOLA",
    "TP_ENSINO", "IN_TREINEIRO", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_PROVA", "TP_LINGUA",
    *[f"Q{i:03d}" for i in range(1, 26)],
    "RESPONSAVEIS_ESCOLARIDADE_IGUAL", "UF_ESCOLA_PROVA_IGUAL",
]
NUM_FEATURES = ["QTD_QUESTIONARIO_PREENCHIDO", "QTD_PERFIL_PREENCHIDO"]
USECOLS = [ID, TARGET, *[c for c in CAT_FEATURES if c.startswith("TP_") or c.startswith("IN_") or c.startswith("SG_") or c.startswith("Q")], "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC"]
USECOLS = list(dict.fromkeys(USECOLS))


def load_data(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    dtype = {c: "category" for c in USECOLS if c not in (ID, TARGET)}
    dtype[ID] = "string"
    dtype[TARGET] = "float32"
    df = pd.read_csv(csv_path, sep=";", encoding="latin1", usecols=USECOLS,
                     dtype=dtype, low_memory=False, na_values=["", " "])
    total = len(df)
    target_observed = df[TARGET].notna()
    eligible = df.loc[target_observed].copy()
    ids = eligible[ID].astype("string").to_numpy()
    y = eligible[TARGET].to_numpy(dtype=np.float32, copy=True)
    raw_cat = [c for c in CAT_FEATURES if c in eligible.columns and c not in ("RESPONSAVEIS_ESCOLARIDADE_IGUAL", "UF_ESCOLA_PROVA_IGUAL")]
    X = eligible[raw_cat].copy()
    q_cols = [f"Q{i:03d}" for i in range(1, 26)]
    present_q = [c for c in q_cols if c in X.columns]
    X["QTD_QUESTIONARIO_PREENCHIDO"] = X[present_q].notna().sum(axis=1).astype("float32")
    profile_cols = [c for c in ["TP_FAIXA_ETARIA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA",
        "TP_NACIONALIDADE", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_ESCOLA", "TP_ENSINO",
        "IN_TREINEIRO", "SG_UF_ESC", "SG_UF_PROVA", "TP_LINGUA"] if c in X.columns]
    X["QTD_PERFIL_PREENCHIDO"] = X[profile_cols].notna().sum(axis=1).astype("float32")
    X["RESPONSAVEIS_ESCOLARIDADE_IGUAL"] = X["Q001"].astype("string").eq(X["Q002"].astype("string")).where(
        X["Q001"].notna() & X["Q002"].notna(), pd.NA).astype("category")
    X["UF_ESCOLA_PROVA_IGUAL"] = X["SG_UF_ESC"].astype("string").eq(X["SG_UF_PROVA"].astype("string")).where(
        X["SG_UF_ESC"].notna() & X["SG_UF_PROVA"].notna(), pd.NA).astype("category")
    for c in CAT_FEATURES:
        if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    info = {"registros_csv": total, "alvo_observado": int(target_observed.sum()),
            "alvo_ausente": int((~target_observed).sum()), "zeros_no_alvo": int(np.count_nonzero(y == 0)),
            "features_categoricas": [c for c in CAT_FEATURES if c in X.columns],
            "features_numericas": NUM_FEATURES, "target": TARGET, "encoding": "latin1 / separador ;"}
    return X, y, {**info, "ids": ids}


def make_preprocessor(X: pd.DataFrame, mode: str = "onehot") -> ColumnTransformer:
    cat = [c for c in CAT_FEATURES if c in X.columns]
    num = [c for c in NUM_FEATURES if c in X.columns]
    if mode == "ordinal":
        # Ambos os encoders representam ausências diretamente como categoria;
        # evita coerções numéricas de pandas.Categorical no SimpleImputer.
        cat_pipe = Pipeline([("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                        encoded_missing_value=-1, dtype=np.float32))])
    else:
        cat_pipe = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                                       sparse_output=True, dtype=np.float32))])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler(with_mean=False))])
    return ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)],
                             remainder="drop", sparse_threshold=1.0)


def build_models(X: pd.DataFrame, expensive: bool = False) -> dict[str, Pipeline]:
    # Preprocessing is fitted inside each estimator (and inside CV when --cv is used).
    def pipe(model, mode="onehot"):
        return Pipeline([("prep", make_preprocessor(X, mode)), ("model", model)])

    models = {
        # O baseline não precisa ler/codificar preditores.
        "baseline_media": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "regressao_linear": pipe(LinearRegression()),
        "ridge": pipe(Ridge(alpha=10.0, solver="lsqr")),
        "svr_linear": pipe(LinearSVR(C=0.1, dual="auto", max_iter=1000, random_state=SEED)),
        "arvore_decisao": pipe(DecisionTreeRegressor(max_depth=18, min_samples_leaf=50, random_state=SEED)),
        "bagging_bootstrap": pipe(BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=12, min_samples_leaf=50),
            n_estimators=5, max_samples=0.1, bootstrap=True, n_jobs=2, random_state=SEED)),
        "pasting_sem_reposicao": pipe(BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=12, min_samples_leaf=50),
            n_estimators=5, max_samples=0.1, bootstrap=False, n_jobs=2, random_state=SEED)),
        "random_forest": pipe(RandomForestRegressor(n_estimators=30, max_depth=18,
            min_samples_leaf=50, max_features=0.7, n_jobs=2, random_state=SEED)),
        "gradient_boosting_hist": pipe(HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=False, random_state=SEED), mode="ordinal"),
        "adaboost": pipe(AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=100),
            n_estimators=20, learning_rate=0.05, random_state=SEED)),
    }
    if expensive:
        models["knn_exato"] = pipe(KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=2))
        base = [
            ("ridge", pipe(Ridge(alpha=10.0, solver="lsqr"))),
            ("arvore", pipe(DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=SEED))),
        ]
        models["stacking"] = StackingRegressor(
            estimators=base, final_estimator=Ridge(alpha=10.0), cv=3, n_jobs=1)
    return models


def score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, pred)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y_true, pred)), "R2": float(r2_score(y_true, pred))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true", help="somente carregar, diagnosticar e gravar as partições")
    parser.add_argument("--cv", action="store_true", help="também executa KFold 3x no desenvolvimento; custo alto")
    parser.add_argument("--include-expensive", action="store_true", help="inclui KNN exato e stacking")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Lendo somente o CSV autorizado: {args.csv}", flush=True)
    t0 = time.perf_counter()
    X, y, meta = load_data(args.csv)
    ids = meta.pop("ids")
    print(f"Carregamento/preparo: {time.perf_counter()-t0:.1f}s; elegíveis={len(y):,}; ausentes={meta['alvo_ausente']:,}", flush=True)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    pd.DataFrame({ID: ids[train_idx]}).to_csv(OUT / "ids_desenvolvimento.csv", index=False)
    pd.DataFrame({ID: ids[test_idx]}).to_csv(OUT / "ids_teste.csv", index=False)
    meta.update({"n_desenvolvimento": len(train_idx), "n_teste": len(test_idx), "seed": SEED,
                 "split": "80/20 aleatório; random_state=42", "tamanho_X_bytes": int(X.memory_usage(deep=True).sum())})
    (OUT / "diagnostico.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({k:v for k,v in meta.items() if k != "features_categoricas" and k != "features_numericas"}, indent=2))
        return

    models = build_models(X, args.include_expensive)
    rows, predictions = [], {ID: ids[test_idx], "nota_observada": y[test_idx]}
    X_dev, y_dev = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    cv_rows = []
    if args.cv:
        cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scoring = {"MSE": "neg_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
        for name, model in models.items():
            if name in ("knn_exato", "stacking"):
                warnings.warn(f"CV omitida para {name}: custo multiplicativo proibitivo na base integral.")
                continue
            start = time.perf_counter()
            res = cross_validate(model, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=1,
                                 return_train_score=False, error_score="raise")
            for fold in range(3):
                mse = -float(res["test_MSE"][fold])
                cv_rows.append({"modelo": name, "dobra": fold+1, "MSE": mse,
                                "RMSE": float(np.sqrt(mse)), "MAE": -float(res["test_MAE"][fold]),
                                "R2": float(res["test_R2"][fold]), "segundos_ajuste": float(res["fit_time"][fold])})
            print(f"CV concluída: {name} ({time.perf_counter()-start:.1f}s)", flush=True)
        pd.DataFrame(cv_rows).to_csv(OUT / "metricas_cv.csv", index=False)

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_dev, y_dev)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test).astype(np.float32, copy=False)
        pred_s = time.perf_counter() - start
        rows.append({"modelo": name, **score(y_test, pred), "segundos_ajuste": fit_s,
                     "segundos_previsao_teste": pred_s})
        predictions[f"prev_{name}"] = pred
        print(f"{name}: RMSE={rows[-1]['RMSE']:.4f}, MAE={rows[-1]['MAE']:.4f}, R2={rows[-1]['R2']:.6f} ({fit_s:.1f}s)", flush=True)
        # Manter artefatos de modelo sem duplicar pipelines intermediários.
        import joblib
        joblib.dump(model, OUT / f"modelo_{name}.joblib", compress=1)

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(OUT / "metricas_teste.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "previsoes_teste.csv", index=False)
    best = {
        "maior_R2": metrics.loc[metrics["R2"].idxmax(), ["modelo", "R2", "MSE", "RMSE", "MAE"]].to_dict(),
        "menor_MSE": metrics.loc[metrics["MSE"].idxmin(), ["modelo", "R2", "MSE", "RMSE", "MAE"]].to_dict(),
        "menor_RMSE": metrics.loc[metrics["RMSE"].idxmin(), ["modelo", "R2", "MSE", "RMSE", "MAE"]].to_dict(),
        "menor_MAE": metrics.loc[metrics["MAE"].idxmin(), ["modelo", "R2", "MSE", "RMSE", "MAE"]].to_dict(),
        "aviso": "Vencedores por métrica são descritivos do conjunto de teste; não usar o teste para seleção/ajuste.",
    }
    (OUT / "melhores_por_metrica.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nClassificação no teste (RMSE menor é melhor):", flush=True)
    print(metrics[["modelo", "MSE", "RMSE", "MAE", "R2"]].to_string(index=False), flush=True)
    print("\nMelhores por métrica (descrição do teste; sem seleção de modelo):", flush=True)
    for metric, item in best.items():
        if isinstance(item, dict):
            print(f"{metric}: {item['modelo']} — R2={item['R2']:.6f}, MSE={item['MSE']:.4f}, RMSE={item['RMSE']:.4f}, MAE={item['MAE']:.4f}", flush=True)
    if args.cv:
        print("\nAtenção: use metricas_cv.csv para selecionar pelo desenvolvimento; o teste é apenas avaliação final.")


if __name__ == "__main__":
    main()

"""Figuras do pitch (docs/pitch.pdf), um gráfico por figura e letras grandes, para cada slide ter um gráfico legível.

Uso, a partir da raiz do projeto: python src/figuras_pitch_graficos.py
Lê resultados/experimentos.csv, resultados/tabelas/*.csv, resultados/previsoes_teste.csv e, só para a distribuição da nota e a
variância do PCA, dados/processed/dataset_1_fundido.csv (desenvolvimento). Grava resultados/figuras/pitch_g_*.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG, TAB = "resultados/figuras/", "resultados/tabelas/"
AZUL, VERM, CINZA, LARANJA, VERDE = "#1f77b4", "#d62728", "#9a9a9a", "#ff7f0e", "#2ca02c"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False})
BR = lambda x, d=0: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
exp = pd.read_csv("resultados/experimentos.csv", index_col="id")


def salvar(fig, nome, fonte="Fonte: Inep, ENEM 2023."):
    fig.text(0.01, 0.005, fonte, fontsize=9, color="#555")
    fig.savefig(FIG + nome, dpi=150)
    plt.close(fig)
    print(nome)


# ---------------------------------------------------------------- desenvolvimento (nota e variância do PCA)
ids = pd.read_csv("dados/processed/ids_elegiveis.csv")["NU_INSCRICAO"]
em_dev = ids.isin(pd.read_csv("dados/splits/desenvolvimento_ids.csv")["NU_INSCRICAO"]).to_numpy()
dev = pd.read_csv("dados/processed/dataset_1_fundido.csv")[em_dev]
nota = dev["NU_NOTA_MT"].to_numpy()

# 1) distribuição da nota
fig, ax = plt.subplots(figsize=(9.02, 3.77))
pos = nota[nota > 0]
ax.hist(pos, bins=60, color=AZUL, label=f"nota > 0 ({BR(len(pos))})")
ax.bar([0], [(nota == 0).sum()], width=14, color=VERM, label=f"nota 0 ({BR((nota == 0).sum())}; {BR((nota == 0).mean() * 100, 2)}%)")
ax.set_xlabel("nota de Matemática")
ax.set_ylabel("participantes")
ax.set_title(f"Nota de Matemática no desenvolvimento ({BR(len(nota))} participantes)")
ax.legend(loc="upper right")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_distribuicao.png")

# 4) variância acumulada do PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(dev.drop(columns="NU_NOTA_MT"))
acum = np.cumsum(PCA(random_state=42).fit(X).explained_variance_ratio_) * 100
k80 = int(np.argmax(acum >= 80)) + 1
fig, ax = plt.subplots(figsize=(9.02, 3.77))
ax.plot(range(1, len(acum) + 1), acum, "o-", color=AZUL)
ax.axhline(80, color=VERM, ls="--", lw=1.3)
ax.axvline(k80, color=VERM, ls=":", lw=1.3)
ax.annotate(f"{k80} componentes = 80% da variância", (k80, 80), xytext=(k80 + 2, 62), fontsize=15, arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("número de componentes")
ax.set_ylabel("variância acumulada (%)")
ax.set_title(f"PCA: variância explicada acumulada ({X.shape[1]} colunas)")
ax.set_ylim(0, 102)
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_pca_variancia.png", "Fonte: Inep, ENEM 2023; desenvolvimento, colunas padronizadas.")

# ---------------------------------------------------------------- modelos
ini = [("Baseline", "H_inicial_Baseline"), ("KNN", "H_inicial_KNN"), ("Regressão linear", "H_inicial_Regressão linear"), ("Ridge", "H_inicial_Ridge"),
       ("SVR linear", "H_inicial_SVR linear"), ("Árvore de decisão", "H_inicial_Árvore de decisão"), ("Bagging", "H_inicial_Bagging"),
       ("Pasting", "H_inicial_Pasting"), ("Random Forest", "H_inicial_Random Forest"), ("Gradient Boosting", "H_inicial_Gradient Boosting"),
       ("Stacking", "H_inicial_Stacking")]
fig, ax = plt.subplots(figsize=(9.02, 4.02))
ys = range(len(ini))
v = [exp.loc[i, "RMSE_media"] for _, i in ini]
d = [exp.loc[i, "RMSE_desvio"] for _, i in ini]
ax.barh(list(ys), v, xerr=d, color=[CINZA if n == "Baseline" else AZUL for n, _ in ini], error_kw=dict(lw=1.5))
for y, vv in zip(ys, v):
    ax.text(vv + 0.4, y, BR(vv, 1), va="center", fontsize=12)
ax.set_yticks(list(ys), [n for n, _ in ini])
ax.set_xlim(105, 136)
ax.set_xlabel("RMSE médio (pontos; menor é melhor)")
ax.set_title("Validação cruzada, configuração inicial (30 colunas)")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_modelos_inicial.png", "Fonte: Inep, ENEM 2023; validação cruzada no desenvolvimento (dataset_1_fundido). Barra de erro: desvio entre dobras.")

# 3) antes e depois da busca
tres = [("Gradient Boosting", "H_inicial_Gradient Boosting", "H_busca_Gradient Boosting"), ("Bagging", "H_inicial_Bagging", "H_busca_Bagging"),
        ("Random Forest", "H_inicial_Random Forest", "H_busca_Random Forest")]
fig, ax = plt.subplots(figsize=(9.02, 3.77))
x = np.arange(3)
a = [exp.loc[i, "RMSE_media"] for _, i, _ in tres]
b = [exp.loc[j, "RMSE_media"] for _, _, j in tres]
ax.bar(x - 0.2, a, 0.4, color=CINZA, label="antes da busca")
ax.bar(x + 0.2, b, 0.4, color=AZUL, label="depois da busca")
for xi, aa, bb in zip(x, a, b):
    ax.text(xi - 0.2, aa + 0.06, BR(aa, 2), ha="center", fontsize=12)
    ax.text(xi + 0.2, bb + 0.06, f"{BR(bb, 2)}\n({bb - aa:+.2f})".replace(".", ","), ha="center", fontsize=12)
ax.set_xticks(x, [n for n, _, _ in tres])
ax.set_ylim(106, 110.9)
ax.set_ylabel("RMSE médio (pontos)")
ax.set_title("Os 3 melhores regressores distintos: antes e depois da busca")
ax.legend(loc="upper left")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_busca.png", "Fonte: Inep, ENEM 2023; RandomizedSearchCV, 10 configurações, validação cruzada no desenvolvimento.")

# 5) PCA: RMSE por número de componentes
fig, ax = plt.subplots(figsize=(9.02, 3.77))
for nome, cor in (("Regressão linear", VERM), ("Gradient Boosting", AZUL)):
    ks = [5, 10, 15, 20, 25]
    y = [exp.loc[f"H_pca_{nome}_{k}", "RMSE_media"] for k in ks] + [exp.loc[f"H_pca_{nome}_sem PCA", "RMSE_media"]]
    ax.plot(ks + [30], y, "o-", color=cor, label=nome)
    ax.text(30.7, y[-1], f"sem PCA:\n{BR(y[-1], 1)}", fontsize=13, color=cor, va="center")
ax.set_xticks([5, 10, 15, 20, 25, 30], ["5", "10", "15", "20", "25", "30"])
ax.set_xlim(3, 38)
ax.set_xlabel("componentes mantidos (30 = todas as colunas, sem PCA)")
ax.set_ylabel("RMSE médio (pontos)")
ax.set_title("PCA: nenhum corte supera o modelo sem PCA")
ax.legend(loc="upper right")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_pca_rmse.png", "Fonte: Inep, ENEM 2023; PCA dentro do Pipeline, validação cruzada no desenvolvimento.")

# ---------------------------------------------------------------- hipóteses e blocos
bl = pd.read_csv(TAB + "blocos_de_informacao.csv", index_col=0).drop(index="completa").sort_values("sobe_RMSE")
fig, ax = plt.subplots(figsize=(9.02, 3.77))
ax.barh(bl.index, bl["sobe_RMSE"], color=AZUL)
for y, v in enumerate(bl["sobe_RMSE"]):
    ax.text(v + 0.05, y, f"+{BR(v, 2)}", va="center", fontsize=13)
ax.axvline(0.19, color=VERM, ls="--", lw=1.3)
ax.text(0.25, 5.62, "linha: desvio entre dobras (0,19)", color=VERM, fontsize=12)
ax.set_ylim(-0.6, 6.0)
ax.set_xlim(0, 5.2)
ax.set_xlabel("aumento do RMSE ao retirar o bloco (pontos)")
ax.set_title("Quanto vale cada bloco de informação")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_blocos.png", "Fonte: Inep, ENEM 2023; Gradient Boosting, validação cruzada; um bloco retirado por vez.")

# 6) balanceamento por faixa (RMSE)
bf = pd.read_csv(TAB + "balanceamento_por_faixa.csv", index_col=0)
bf = bf.drop(index="0")
fig, ax = plt.subplots(figsize=(9.02, 3.77))
x = np.arange(len(bf))
for k, (col, nome, cor) in enumerate((("RMSE sem pesos", "sem pesos", AZUL), ("RMSE α = 0,5", "α = 0,5", LARANJA), ("RMSE α = 1,0", "α = 1,0", VERDE))):
    ax.bar(x + (k - 1) * 0.27, bf[col], 0.27, color=cor, label=nome)
ax.set_xticks(x, bf.index)
ax.set_xlabel("faixa de nota real (a faixa 0, com RMSE em torno de 510, não aparece)")
ax.set_ylabel("RMSE (pontos)")
ax.set_title("RMSE por faixa de nota, com e sem pesos")
ax.set_ylim(0, 235)
ax.legend(loc="upper left")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_balanceamento.png", "Fonte: Inep, ENEM 2023; validação cruzada no desenvolvimento; pesos n^(−α) por faixa de 50 pontos, calculados só no treino.")

# 7) dataset 6: RMSE do Gradient Boosting
dg = pd.read_csv(TAB + "diagnostico_dataset6_notas_outras_areas.csv")
dg = dg[dg["modelo"] == "Gradient Boosting"].reset_index(drop=True)
rot = ["só o perfil\n(45 colunas)", "+ Natureza\n(mesmo dia)", "+ Humanas,\nLinguagens e\nRedação (dia 1)", "+ as quatro\nnotas"]
fig, ax = plt.subplots(figsize=(9.02, 3.77))
ax.bar(range(len(dg)), dg["RMSE"], color=[AZUL] + [VERM] * 3)
for i, r in dg.iterrows():
    ax.text(i, r["RMSE"] + 1.5, f"RMSE {BR(r['RMSE'], 1)}\nR² {BR(r['R2'], 2)}", ha="center", fontsize=14)
ax.set_xticks(range(len(dg)), rot)
ax.set_ylim(0, 145)
ax.set_ylabel("RMSE médio (pontos)")
ax.set_title("Gradient Boosting com e sem as notas de outras áreas (vazamento)")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_dataset6.png", "Fonte: Inep, ENEM 2023; dataset 6, só validação cruzada no desenvolvimento; fora da solução.")

# ---------------------------------------------------------------- solução final
cu = pd.read_csv(TAB + "curva_aprendizado_resumo.csv")
fig, ax = plt.subplots(figsize=(9.02, 3.77))
ax.errorbar(cu["n_treino"] / 1e6, cu["treino"], cu["treino_dp"], fmt="o-", color=AZUL, lw=2, ms=8, capsize=4, label="treino")
ax.errorbar(cu["n_treino"] / 1e6, cu["validacao"], cu["validacao_dp"], fmt="o-", color=LARANJA, lw=2, ms=8, capsize=4, label="validação")
ax.axhline(131.63, color=VERM, ls="--", lw=1.2, label="baseline (131,6)")
ax.set_ylim(100, 134)
ax.set_xlabel("linhas de treino (milhões)")
ax.set_ylabel("RMSE (pontos)")
ax.set_title("Curva de aprendizado da solução final")
ax.legend(loc="center right")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_curva.png", "Fonte: Inep, ENEM 2023; validação cruzada no desenvolvimento; média e desvio das 3 dobras.")

# ---------------------------------------------------------------- teste
pt = pd.read_csv("resultados/previsoes_teste.csv")
obs, prev = pt["nota_observada"].to_numpy(), pt["previsao_final"].to_numpy()
fig, ax = plt.subplots(figsize=(9.02, 4.02))
hb = ax.hexbin(obs, prev, gridsize=70, bins="log", mincnt=1, cmap="viridis", extent=(0, 1000, 0, 1000))
ax.plot([0, 1000], [0, 1000], color=VERM, lw=2, label="previsão perfeita")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_xlabel("nota observada")
ax.set_ylabel("nota prevista")
ax.set_title(f"Teste: nota observada contra prevista ({BR(len(obs))} participantes)")
ax.legend(loc="upper left")
fig.colorbar(hb, ax=ax, label="participantes (escala log)")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_obs_prev.png", "Fonte: Inep, ENEM 2023; teste; dispersão agregada em células hexagonais.")

res = obs - prev
fig, ax = plt.subplots(figsize=(9.02, 4.02))
hb = ax.hexbin(prev, res, gridsize=70, bins="log", mincnt=1, cmap="viridis")
ax.axhline(0, color=VERM, lw=2, label=f"resíduo médio ({BR(res.mean(), 1)})")
ax.set_xlabel("nota prevista")
ax.set_ylabel("resíduo (observada − prevista)")
ax.set_title("Teste: resíduo contra nota prevista")
ax.legend(loc="upper right")
fig.colorbar(hb, ax=ax, label="participantes (escala log)")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_resid.png", "Fonte: Inep, ENEM 2023; teste; a faixa de pontos inclinada, abaixo de −370, são as notas zero.")

g = pd.read_csv(TAB + "teste_erros_por_grupo.csv", index_col=0)
geral = 105.81
t = g[g["tipo"] == "faixa de nota"]
fig, ax = plt.subplots(figsize=(9.02, 3.77))
ax.bar(range(len(t)), t["RMSE"], color=[VERM if v > 150 else AZUL for v in t["RMSE"]])
for i, (n, v) in enumerate(zip(t["n"], t["RMSE"])):
    ax.text(i, v + 8, f"{BR(v, 0)}\nn={BR(n)}", ha="center", fontsize=12)
ax.axhline(geral, color="black", ls="--", lw=1.2)
ax.text(len(t) - 0.5, 330, "linha tracejada: RMSE geral (105,81)", ha="right", fontsize=13)
ax.set_xticks(range(len(t)), t.index)
ax.set_ylim(0, 620)
ax.set_xlabel("faixa da nota observada")
ax.set_ylabel("RMSE (pontos)")
ax.set_title("Erro por faixa de nota (teste): o modelo erra mais nas pontas")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_erros_faixa.png", "Fonte: Inep, ENEM 2023; teste (538.486 participantes); faixas definidas no desenvolvimento.")

linhas = []
for tipo, nome in (("região da prova", "Região"), ("renda familiar", "Renda"), ("tipo de escola", "Escola")):
    for gr, r in g[g["tipo"] == tipo].iterrows():
        linhas.append((f"{nome}: {gr}", r["RMSE"], r["n"], nome))
fig, ax = plt.subplots(figsize=(9.02, 4.02))
cores = {"Região": AZUL, "Renda": LARANJA, "Escola": VERDE}
for y, (rot, v, n, grp) in enumerate(linhas):
    ax.barh(y, v, color=cores[grp])
    ax.text(v + 1, y, f"{BR(v, 0)}  (n={BR(n)})", va="center", fontsize=12)
ax.set_yticks(range(len(linhas)), [r[0] for r in linhas], fontsize=12)
ax.invert_yaxis()
ax.axvline(geral, color="black", ls="--", lw=1.2)
ax.text(geral + 1, -0.85, "RMSE geral (105,81)", fontsize=12)
ax.set_ylim(len(linhas) - 0.4, -1.3)
ax.set_xlim(0, 150)
ax.set_xlabel("RMSE (pontos)")
ax.set_title("Erro por região, renda e tipo de escola (teste)")
plt.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "pitch_g_erros_grupos.png", "Fonte: Inep, ENEM 2023; teste (538.486 participantes); n = participantes do grupo.")
print("ok")

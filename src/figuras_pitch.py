"""Figuras dos slides 5 e 6 (docs/pitch.pdf): ausentes na base bruta e features criadas. As demais estão em src/figuras_pitch_graficos.py.

Uso, a partir da raiz do projeto: python src/figuras_pitch.py
Lê o CSV bruto do Inep (colunas escolhidas). Grava em resultados/figuras/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = "resultados/figuras/"
FONTE = "Fonte: Inep, ENEM 2023."

# 1) valores ausentes (base bruta completa, 3.933.955 registros)
cols = ["NO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "TP_ENSINO",
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
bruto = pd.read_csv("dados/raw/DADOS/MICRODADOS_ENEM_2023.csv", sep=";", encoding="latin1", usecols=cols)
aus = (bruto.isna().mean() * 100).sort_values()
cor = ["tab:red" if c.startswith("NU_NOTA") else "tab:orange" for c in aus.index]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.barh(aus.index, aus.values, color=cor)
for i, v in enumerate(aus.values):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=12)
ax.set_xlim(0, 100)
ax.set_xlabel("% de valores ausentes na base bruta")
ax.set_title("Mais ausência (laranja: escola; vermelho: notas)", fontsize=13)
ax.tick_params(axis="y", labelsize=12)
fig.text(0.01, 0.01, FONTE + " 3.933.955 registros.", fontsize=8)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(FIG + "pitch_ausentes.png", dpi=150)
plt.close()

# 2) features criadas: Spearman com a nota (notebook 01, seção 11c)
feat = {"PESSOAS_POR_QUARTO\n(casa lotada, menos espaço de estudo)": -0.221,
        "CELULAR_POR_MORADOR\n(mais acesso próprio a conteúdo)": 0.280,
        "RENDA_MENOS_ESCOL\n(renda alta ou baixa para o nível de estudo)": 0.090}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.barh(list(feat)[::-1], list(feat.values())[::-1], color=["tab:green" if v > 0 else "tab:red" for v in list(feat.values())[::-1]])
for i, v in enumerate(list(feat.values())[::-1]):
    ax.text(v + (0.012 if v > 0 else -0.012), i, f"{v:+.2f}".replace(".", ","), va="center", ha="left" if v > 0 else "right", fontsize=14)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.36, 0.40)
ax.set_xlabel("Spearman com a nota de Matemática")
ax.set_title("Features criadas e a hipótese de cada uma", fontsize=13)
ax.tick_params(axis="y", labelsize=12)
fig.text(0.01, 0.01, FONTE + " Desenvolvimento; todas com |Spearman| < 0,7 entre si e com as colunas originais.", fontsize=8)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(FIG + "pitch_features_criadas.png", dpi=150)
plt.close()

print("ok")

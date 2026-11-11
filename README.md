# Projeto 2 - Ciência de Dados: Previsão da nota de Matemática do ENEM 2023

Projeto da disciplina de Ciência de Dados (Unifor, Unidade II – Aprendizado de Máquina). Investiga **quanto o perfil socioeconômico, escolar e demográfico dos participantes permite estimar a nota de Matemática do ENEM 2023**, comparando escolhas de preparação e de modelagem e explicando os resultados e os limites das previsões.

> **Etapa atual: entrega final.** O conjunto de teste foi aberto **uma única vez**, no fim do notebook `04-avaliacao-final.ipynb`, com a solução já fixada pela validação cruzada do desenvolvimento. Nenhuma escolha de feature, técnica ou hiperparâmetro usou o teste.

Documentos: documentação completa em [`docs/documentacao_completa.docx`](docs/documentacao_completa.docx) (24 seções e apêndices, as 87 figuras e as tabelas de todos os experimentos, a tabela dos 82 experimentos com métricas por dobra e as tabelas com os valores de Spearman de cada figura), artigo em [`docs/artigo.pdf`](docs/artigo.pdf) (e em LaTeX, [`docs/artigo.tex`](docs/artigo.tex)), slides do pitch em [`docs/pitch.pdf`](docs/pitch.pdf) (25 slides, um gráfico por slide) e relatório técnico dos modelos em [`docs/relatorio_detalhado_modelos.pdf`](docs/relatorio_detalhado_modelos.pdf).

---

## 👥 Equipe e contribuições

| Integrante | Contribuições no projeto |
| :--- | :--- |
| João Igor Vidal de Andrade | Exploração complementar dos dados e treinamento preliminar do modelo. Responsável pelo treinamento dos modelos: baseline pela média e Ridge. |
| Raquel Albuquerque Quirino | Criação dos documentos, dos slides e do artigo. Organização das tarefas da equipe. |
| Amanda Lira | Análise exploratória dos dados, limpeza e higienização da base, análise de outliers e demais etapas de preparação dos dados. |
| Sofya Oliveira | Engenharia de features e análise da relação das features com o alvo, para orientar as escolhas do grupo. |

---

## 🎯 Alvo, tarefa e fonte dos dados

| | |
| :--- | :--- |
| **Alvo** | `NU_NOTA_MT` – nota de Matemática e suas Tecnologias (escala de 0 a 1000). |
| **Tarefa** | Regressão supervisionada: o alvo é um valor numérico contínuo, não uma classe. |
| **Entradas (X)** | Perfil demográfico, trajetória escolar, escola, UF da prova e questionário socioeconômico, todos conhecidos antes da prova. |
| **Fonte** | Inep – [Microdados do ENEM 2023](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem), base nacional completa (3.933.955 registros e 76 colunas). |
| **Repositório** | https://github.com/Alyra004/ciencia-de-dados-unifor-projeto-2 |

---

## 📐 Protocolo de avaliação

* **Partição:** participantes com nota observada divididos aleatoriamente em 80% para desenvolvimento e 20% para teste, com `random_state=42`. Os identificadores ficam em `dados/splits/` e não são entradas do modelo.
* **Validação cruzada:** só no desenvolvimento, com `KFold(n_splits=3, shuffle=True, random_state=42)` e as mesmas dobras em todas as comparações (`dados/splits/dobras_cv.csv`). Não existe uma segunda parcela de validação.
* **Baseline:** `DummyRegressor(strategy="mean")`, com a média recalculada no treino de cada dobra (e, no teste, nos 80% de desenvolvimento).
* **Métricas:** RMSE (principal), MAE e R², na escala original da nota, com média e desvio padrão entre as dobras.
* **Sem vazamento entre dobras:** tudo o que é aprendido dos dados (fusão da situação escolar, padronização, seleção de atributos, PCA, pesos por faixa) fica dentro de um `Pipeline` ou é calculado só com o treino de cada dobra.
* **Teste reservado:** usado uma vez, depois de fixar a solução.

---

## 📊 Dados e decisões de preparação

| Indicador | Quantidade |
| :--- | ---: |
| Registros na base bruta | 3.933.955 |
| Sem nota de Matemática (excluídos da regressão, 31,56%) | 1.241.528 |
| **Com nota de Matemática observada (elegíveis, 68,44%)** | **2.692.427** |
| Desenvolvimento (80%) | 2.153.941 |
| Teste (20%) | 538.486 |

* A nota ausente **não é imputada nem convertida em zero**. A nota **zero** é um resultado observado (13.300 casos no desenvolvimento, 0,62%) e fica no treino e na validação. Treineiros são mantidos. Não há amostra por conveniência: todas as linhas elegíveis são usadas.
* **Fora de X** ficam as notas de outras áreas e da redação, `TP_PRESENCA_*`, `TP_STATUS_REDACAO`, `TX_RESPOSTAS_*`, `TX_GABARITO_*`, `CO_PROVA_*`, `NU_ANO` e o identificador.
* **Codificação (notebook 01):** questionário `Q001`–`Q025` de letras para números (A=0, B=1...); sexo em 0/1; "não sei" em `Q001`–`Q004` vira ausente, é preenchido com a mediana (que é a mesma no desenvolvimento e em todas as linhas) e ganha indicadores `Q00X_NAO_SEI` (quem não sabe tira 40 a 50 pontos a menos); escola, localização da escola e nacionalidade em colunas 0/1 (ausência de escola = "sem informação" = referência). Os datasets salvos não têm nenhum nulo.
* **Features criadas (3, com hipóteses no notebook 01, seção 11c):** `PESSOAS_POR_QUARTO` (moradores por quarto, denominador mínimo 1), `CELULAR_POR_MORADOR` (celulares por morador) e `RENDA_MENOS_ESCOL` (renda e escolaridade dos responsáveis na escala 0–1).
* **Redundância:** Spearman entre colunas, limite |ρ| > 0,7. As quatro colunas de situação escolar eram redundantes e foram fundidas na `FUSAO_1` nos datasets 1 a 3. **No notebook 04 a fusão é refeita só com o treino de cada dobra** (corrige o vazamento da média e do desvio) e é comparada com manter as 4 colunas separadas.
* **8 datasets** em `dados/processed/` (1: fundido; 1 + features; 2: + escola; 2 + features; 3: + `NAO_SEI` e nacionalidade, com e sem escola, com e sem features), comparados no notebook 02.

---

## 🧪 Modelos e resultados

Todos os números abaixo vêm de `resultados/experimentos.csv` (uma linha por experimento, com hipótese, features, preparação, modelo, parâmetros, métricas por dobra, média, desvio, tempo e conclusão). A validação cruzada do desenvolvimento é separada do teste.

### 1. Configuração inicial dos modelos (notebook 02, `dataset_1_fundido`, validação cruzada)

| Modelo | RMSE (média ± desvio) | MAE | R² | Tempo/dobra (s) |
| --- | ---: | ---: | ---: | ---: |
| Stacking | 107,763 ± 0,226 | 83,804 | 0,330 | 629,8 |
| Gradient Boosting | 107,770 ± 0,226 | 83,825 | 0,330 | 132,7 |
| Random Forest | 109,422 ± 0,195 | 85,494 | 0,309 | 103,4 |
| Pasting | 110,012 ± 0,208 | 85,996 | 0,302 | 19,8 |
| Bagging | 110,014 ± 0,207 | 85,991 | 0,302 | 24,1 |
| Árvore de decisão | 110,551 ± 0,190 | 86,381 | 0,295 | 3,0 |
| Regressão linear | 111,153 ± 0,178 | 86,960 | 0,287 | 0,7 |
| Ridge | 111,153 ± 0,178 | 86,960 | 0,287 | 0,3 |
| SVR linear | 111,153 ± 0,178 | 86,961 | 0,287 | 1,9 |
| KNN | 111,575 ± 0,184 | 87,180 | 0,282 | 476,3 |
| Baseline | 131,634 ± 0,158 | 107,582 | -0,000 | 0,1 |

### 2. Os 3 melhores regressores distintos, antes e depois da busca (`RandomizedSearchCV`, 10 configurações, 3 dobras)

| Regressor | RMSE antes | RMSE depois | Ganho | R² depois | Tempo/dobra (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 107,770 | 107,680 ± 0,178 | 0,090 | 0,331 | 224 |
| Bagging | 110,014 | 109,083 ± 0,154 | 0,931 | 0,313 | 24 |
| Random Forest | 109,422 | 109,138 ± 0,159 | 0,284 | 0,313 | 19 |

### 3. Hipóteses testadas no notebook 04 (Gradient Boosting inicial, validação cruzada)

| Experimento | RMSE (média ± desvio) | Conclusão automática |
| --- | ---: | ---: |
| 1a. Fusão da situação escolar refeita só com o treino | 107,051 ± 0,192 | referência |
| 1b. 4 colunas de situação separadas | 106,726 ± 0,189 | RMSE -0.325 em relação a S1a_fusao_no_treino: melhora (maior que o desvio entre dobras, 0.192). |
| 2. + UF da prova (26 colunas 0/1) | 106,025 ± 0,188 | RMSE -0.701 em relação a S1b_colunas_separadas: melhora (maior que o desvio entre dobras, 0.189). |
| 3. Sem as 3 features criadas | 106,038 ± 0,167 | RMSE +0.013 em relação a S2_com_uf: sem diferença clara (dentro do desvio entre dobras, 0.188). |
| 4a. SelectKBest k = 51 | 107,069 ± 0,199 | RMSE +1.044 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 4b. SelectKBest k = 37 | 107,436 ± 0,208 | RMSE +1.412 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 4c. SelectKBest k = 22 | 110,607 ± 0,191 | RMSE +4.582 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 5a. Pesos por faixa α = 0,5 | 108,140 ± 0,188 | RMSE +2.115 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 5b. Pesos por faixa α = 1,0 | 123,459 ± 0,149 | RMSE +17.434 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6a. Sem perfil e trajetória | 110,445 ± 0,189 | RMSE +4.420 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6b. Sem bens e domicílio | 107,356 ± 0,174 | RMSE +1.331 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6c. Sem renda | 106,893 ± 0,194 | RMSE +0.868 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6d. Sem UF | 106,726 ± 0,189 | RMSE +0.701 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6e. Sem escolaridade e ocupação dos responsáveis | 106,559 ± 0,168 | RMSE +0.534 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |
| 6f. Sem escola | 106,439 ± 0,185 | RMSE +0.414 em relação a S2_com_uf: piora (maior que o desvio entre dobras, 0.188). |

Regra de leitura: só chamamos de melhora ou piora se a diferença do RMSE médio passa do desvio entre dobras da referência.

### 4. Solução final

**Solução final:** `HistGradientBoostingRegressor` com os hiperparâmetros da busca (`learning_rate` 0,067, até 500 árvores, 89 folhas, `min_samples_leaf` 137, `l2_regularization` 0,244, parada antecipada com 10% do treino) sobre **74 colunas** sem nulos: 44 do dataset 3 (sem a `FUSAO_1`), as 4 colunas de situação escolar separadas e 26 de UF da prova. Sem seleção de atributos, sem PCA e sem pesos por faixa (cada um foi testado e não adotado). No teste, o RMSE é 105,81 contra 131,70 do baseline (19,7% menos erro) e 108,86 da regressão linear; o teste confirma a validação (105,86 ± 0,17).

| Modelo | CV RMSE (média ± desvio) | CV MAE | CV R² | Teste RMSE | Teste MAE | Teste R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (média) | 131,63 ± 0,16 | 107,58 ± 0,06 | 0,000 | 131,70 | 107,64 | 0,000 |
| Regressão linear (sem regularização) | 108,87 ± 0,16 | 84,86 ± 0,04 | 0,316 | 108,86 | 84,89 | 0,317 |
| **Solução final (Gradient Boosting)** | 105,86 ± 0,17 | 82,02 ± 0,05 | 0,353 | 105,81 | 82,00 | 0,354 |

* **Curva de aprendizado** (10, 25, 50, 75 e 100% do treino de cada dobra): `resultados/tabelas/curva_aprendizado*.csv` e `resultados/figuras/curva_aprendizado.png`.
* **Erros por faixa de nota e por grupo** (região e UF da prova, renda, tipo de escola, sexo, com os tamanhos dos grupos): `resultados/tabelas/teste_erros_por_grupo.csv`.
* **Previsões do teste:** `resultados/previsoes_teste.csv` (identificador, nota observada e previsões do baseline, da regressão linear e da solução final).

### Outros resultados

* **Dataset 5 (replicação independente, solução de outra IA, pasta `resultados/dataset5_replicacao_independente/`):** treino feito só com o enunciado e o CSV, no mesmo teste que o nosso. Melhor modelo: Ridge com one-hot de todas as categóricas, RMSE 107,53 no teste (nossa solução: 105,81); o Gradient Boosting leve dela fez 108,07. O script e as métricas dela ficam em `resultados/dataset5_replicacao_independente/`. Avaliado também por validação cruzada nas nossas dobras (notebook 06): Ridge 107,51, SVR linear 107,90, Gradient Boosting leve 107,99 e regressão linear 111,29 (instável); a nossa solução fez 105,86 (1,6 ponto melhor). Detalhes na seção 8.4 e no Apêndice G da documentação completa.
* **Dataset 6 (diagnóstico com vazamento, fora da solução):** com as notas de Natureza, Humanas, Linguagens e Redação, o Gradient Boosting vai de RMSE 107,05 (R² 0,339) para 74,43 (R² 0,680) na validação cruzada. Só Natureza (mesmo dia) já dá 82,83. Confirma que o limite de R² ≈ 0,35 vem da informação do perfil. No Spearman (só desenvolvimento), as quatro notas são as colunas mais associadas à nota de Matemática (Humanas 0,66, Linguagens 0,64, Natureza 0,64, Redação 0,53, contra 0,46 da renda). Tirar as notas zero do treino não muda nada com só o perfil, mas **piora muito** com as notas (RMSE de 74,4 para 82,8): 98,3% de quem zerou Matemática também zerou Natureza, então os zeros ficam identificáveis e deixam de ser ruído.

* **Extensões finais** (notebook 06, só validação cruzada; o teste já tinha sido aberto, então nada muda a solução final): (1) one-hot em todas as categóricas reduz o RMSE dos modelos lineares de 108,87 para 107,89 (cerca de 1 ponto; só nas nominais, 0,24); (2) V de Cramér: só situação de conclusão × tipo de escola passa de 0,7 (0,707); (3) município com target encoding dentro do Pipeline: 106,03 → 105,73 (ganho de 0,3, em todas as dobras; não adotado); (4) sementes 0, 1 e 2: 105,862 a 105,871, variação de 0,009; (5) busca de hiperparâmetros nas 74 colunas finais: Gradient Boosting 105,859 (igual aos 105,864 atuais), Bagging 107,88 e Random Forest 107,92; (6) dataset 5 por validação cruzada (acima).
* **PCA** (notebook 03): para 80% da variância são necessários 18 de 30 componentes; em regressão linear e Gradient Boosting, o PCA **piorou** o RMSE em todos os números de componentes testados. t-SNE e K-means servem para visualizar; as distâncias aos grupos do K-means não ajudaram.
* **Notas zero fora do treino** (notebook 02 sem zeros): sem diferença clara no RMSE geral.
* **Balanceamento por faixas** (seção 5 do notebook 04): dar peso às faixas raras reduz o erro nas notas altas (faixa 800+: RMSE de 185 para 155 com α = 0,5), mas piora o erro geral e o das faixas do meio. Não adotado.

---

## 💻 Ambiente

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Resultados gerados com Python 3.14.8, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, em um computador com 10 núcleos e 16 GB de memória. Os estimadores usados rodam em CPU. Outras versões devem funcionar, com diferenças na última casa decimal. Semente 42 em tudo que aceita `random_state`.

---

## 🚀 Ordem de execução

Rode os notebooks a partir da pasta `notebooks/`.

1. **Obter os dados:** baixar os Microdados do ENEM 2023 no site do Inep (link acima) e colocar `MICRODADOS_ENEM_2023.csv` em `dados/raw/DADOS/`. Os arquivos grandes não vão para o GitHub.
2. **`01-exploracao-filtro.ipynb`:** triagem, filtro dos participantes com nota (`ids_elegiveis.csv`), codificação, features criadas e os 8 datasets em `dados/processed/`.
3. **Partição (opcional):** `dados/splits/` já está no repositório. `python src/particao.py` confere que os arquivos são iguais aos gerados com a semente 42 (`--gravar` recria).
4. **`02-treino-modelos.ipynb`:** baseline e todos os regressores obrigatórios em configuração inicial, comparação dos modelos e dos 8 datasets. Modelos em `models/` (fora do git).
5. **`02-treino-modelos-optimized.ipynb`:** `RandomizedSearchCV` (10 configurações) nos regressores; tabelas em `resultados/tabelas/busca_*.csv`.
6. **`02-treino-modelos-sem-zeros-optimized.ipynb`:** treino sem as notas zero e efeito por faixa.
7. **`03-reducao-dimensionalidade.ipynb`:** PCA exploratório e dentro do Pipeline, t-SNE e K-means.
8. **`04-avaliacao-final.ipynb`:** fusão sem vazamento, UF, features criadas, seleção, balanceamento, blocos de informação, solução final, curva de aprendizado e **teste final**. Grava `resultados/experimentos.csv`, `resultados/previsoes_teste.csv`, `resultados/tabelas/` e `resultados/figuras/`.

9. **`05-diagnostico-notas-outras-areas.ipynb` (opcional, fora da solução):** o **dataset 6**, com as notas de outras áreas, mede o tamanho do vazamento que o enunciado proíbe. Só validação cruzada; o teste não é aberto. Grava `resultados/tabelas/diagnostico_dataset6_*.csv` (CV, sem zeros e Spearman), figuras `diagnostico_dataset6*.png` e `dados/processed/dataset_6_notas_outras_areas.csv`.

10. **`06-extensoes-finais.ipynb`:** one-hot nos modelos lineares, V de Cramér, município com target encoding, três sementes, busca de hiperparâmetros na representação final e o dataset 5 por validação cruzada. Só validação cruzada; grava as linhas `E1_*` a `E6_*` do `resultados/experimentos.csv`, `resultados/tabelas/e*_*.csv` e `resultados/figuras/e2_cramer_v.png` (cerca de 2 horas).
11. **`07-metricas-por-dobra-historico.ipynb`:** refaz, com as mesmas configurações, os experimentos dos notebooks 02 e 03 (linhas `H_*`) gravando as métricas das 3 dobras no `experimentos.csv`, e imprime a diferença para a média antiga (cerca de 3 horas). As funções de apoio estão em `src/cv_utils.py`.

`src/modelagem.py` tem a classe `FusaoSituacao` para carregar os modelos salvos fora do notebook (instruções no próprio arquivo). Tempo total aproximado: o notebook 04 leva cerca de 1 hora; o 02 otimizado, mais de 2 horas.

---

## 🗂️ Estrutura do repositório

```
dados/        raw/ (originais, fora do git), processed/ (datasets, fora do git), splits/ (ids das partições e dobras)
notebooks/    01 exploração e datasets, 02 modelos (inicial, otimizado, sem zeros), 03 redução de dimensionalidade, 04 avaliação final, 05 dataset 6 (diagnóstico), 06 extensões finais, 07 métricas por dobra do histórico
src/          particao.py (recria/confere a partição e as dobras), modelagem.py (classe FusaoSituacao do pipeline final), cv_utils.py (validação cruzada por dobra e registro em experimentos.csv), figuras_pitch.py e figuras_pitch_graficos.py (figuras do pitch)
models/       modelos treinados (*.joblib, fora do git)
resultados/   experimentos.csv, previsoes_teste.csv, tabelas/, figuras/, dataset5_replicacao_independente/ (script e métricas da solução independente)
docs/         documentacao_completa.docx, artigo.pdf, artigo.tex, pitch.pdf, relatorio_detalhado_modelos.pdf, documentacao_decisoes.docx
requirements.txt
```

---

## ⚠️ Limitações

* O desempenho é limitado pela **informação**, não pelo modelo: o perfil (renda, escolaridade, bens, escola, UF) explica cerca de 35% da variação da nota (R² de 0,354 no teste). Associações entre perfil e nota **não indicam causa** nem a capacidade individual dos participantes.
* A divisão principal avalia **novos participantes da mesma edição**; não mostra generalização para escolas ou municípios inéditos. UF e escola são da prova e do cadastro, não da residência.
* Os hiperparâmetros foram buscados no `dataset_1_fundido` e reaproveitados na representação final. Uma busca nova nas 74 colunas (notebook 06) deu o mesmo erro (105,859 contra 105,864), então a limitação não pesa.
* No notebook sem zeros, as buscas gravam `busca_sem_zeros_<modelo>.csv`, para não colidir com os `busca_<modelo>.csv` do notebook otimizado.

## 📚 Referências

Inep (Microdados do ENEM 2023), Pedregosa et al. (scikit-learn, JMLR 2011), Friedman (2001, gradient boosting), Breiman (1996, 2001), Wolpert (1992, stacking), Bergstra e Bengio (2012, busca aleatória). Lista completa no artigo (`docs/artigo.pdf`).

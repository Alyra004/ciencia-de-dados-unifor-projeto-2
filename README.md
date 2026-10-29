# Projeto 2 - Ciência de Dados: Previsão da nota de Matemática do ENEM 2023

Projeto da disciplina de Ciência de Dados (Unifor, Unidade II – Aprendizado de Máquina). Investiga **quanto o perfil socioeconômico, escolar e demográfico dos participantes permite estimar a nota de Matemática do ENEM 2023**, comparando escolhas de preparação e de modelagem e explicando os resultados e os limites das previsões.

> **Etapa atual: acompanhamento.** O conjunto de teste está reservado e **não foi usado**. Todos os resultados abaixo são de validação cruzada no desenvolvimento. A avaliação considera a qualidade da investigação, e não um R² mínimo.

O documento completo, com tabelas, figuras e a discussão de cada decisão, está em [`docs/documentacao_decisoes.docx`](docs/documentacao_decisoes.docx). Este README resume o mesmo conteúdo, na mesma ordem.

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
| **Entradas (X)** | Perfil demográfico, trajetória escolar e questionário socioeconômico, todos conhecidos antes da prova. |
| **Fonte** | Inep – [Microdados do ENEM 2023](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem), base nacional completa (3.933.955 registros e 76 colunas). |
| **Repositório** | https://github.com/Alyra004/ciencia-de-dados-unifor-projeto-2 |

---

## 📐 Protocolo de avaliação adotado

* **Partição:** participantes com nota observada divididos aleatoriamente em 80% para desenvolvimento e 20% para teste, com `random_state=42`. Os identificadores ficam em `dados/splits/` e não são entradas do modelo.
* **Validação cruzada:** apenas no desenvolvimento, com `KFold(n_splits=3, shuffle=True, random_state=42)` e as mesmas dobras em todas as comparações. As dobras estão salvas em `dados/splits/dobras_cv.csv`.
* **Baseline:** `DummyRegressor(strategy="mean")`, com a média recalculada no treino de cada dobra.
* **Métricas:** RMSE (principal), MAE e R², sempre na escala original da nota, com média e desvio padrão entre as dobras.
* **Sem vazamento entre dobras:** toda transformação aprendida dos dados (imputação, padronização, seleção, fusão de colunas, PCA) fica dentro de um `Pipeline` e é reajustada apenas no treino de cada dobra.
* **Teste reservado:** só é usado no final, depois de fixar a solução, e nunca para escolher features, técnicas ou hiperparâmetros.

---

## 📊 Dados e decisões de preparação

| Indicador | Quantidade |
| :--- | ---: |
| Registros na base bruta | 3.933.955 |
| Sem nota de Matemática (excluídos da regressão, 31,56%) | 1.241.528 |
| **Com nota de Matemática observada (elegíveis, 68,44%)** | **2.692.427** |
| Desenvolvimento (80%) | 2.153.941 |
| Teste (20%, reservado) | 538.486 |

* A nota ausente **não é imputada nem convertida em zero**. Os treineiros são mantidos. Não foi usada amostra por conveniência.
* **Distribuição do alvo no desenvolvimento:** média 533,83; mediana 523,50; desvio padrão 131,63 (praticamente o RMSE do baseline); 13.300 notas zero (0,617%), mantidas por serem um resultado observado e válido.
* **Prevenção de vazamento:** ficam fora de X todas as notas de outras áreas e da redação, `TP_PRESENCA_*`, `TP_STATUS_REDACAO`, `TX_RESPOSTAS_*`, `TX_GABARITO_*`, `CO_PROVA_*` e `NU_ANO`. Uma verificação automática (`verificar_sem_vazamento`, em `src/config.py`) interrompe a execução se alguma aparecer.
* **Features:** a equipe partiu de 46 colunas candidatas e definiu uma primeira configuração com **33 features** (8 de perfil e trajetória escolar e as 25 do questionário socioeconômico, `Q001` a `Q025`), codificadas em **138 colunas numéricas**. As demais candidatas (município e UF da escola e da prova, dependência e localização da escola, `TP_NACIONALIDADE`, `TP_ANO_CONCLUIU`, `TP_LINGUA`) ficaram fora, por baixa cobertura (73,10% de ausências nas colunas de escola) ou por decisão de escopo, e podem voltar em experimentos próprios.
* **Codificação:** determinística, pelo dicionário de 2023 (nada é aprendido dos dados). Nominais em one-hot; escolaridade em níveis de 0 a 6, com o indicador `DESCONHECE` para "não sei"; renda pelo ponto médio da faixa; contagens de 0 a 4. As ausências ganham indicadores `__AUSENTE` e `__OUTRO`. Das 138 colunas, 66 não variam no desenvolvimento, e só 72 têm variação.
* **Features derivadas (5):** renda per capita, escolaridade máxima dos responsáveis, índice de acesso digital, moradores por quarto e interação renda × escolaridade.
* **Redundância e fusão:** cinco pares de colunas têm correlação de Spearman acima de 0,85. O `FeatureAgglomeration` sugeriu dois grupos (renda e escolaridade), fundidos no conjunto D (5 colunas viram 2).
* **"Não sei" em Q001 e Q002:** 10,23% do desenvolvimento (220.341 registros) e nota média cerca de 41 pontos menor. Os registros são mantidos, com o indicador `DESCONHECE`; remover ou não é uma decisão em aberto.

---

## 🧪 Modelos e resultados até agora

Três conjuntos de features, com as mesmas dobras: **A** (138 colunas originais tratadas), **B** (A + 5 derivadas, 143 colunas) e **D** (B com os grupos redundantes fundidos, 140 colunas); o **C** é B reduzida a 50 colunas por `SelectKBest`. Validação cruzada de 3 dobras no desenvolvimento (média ± desvio):

| Experimento | RMSE | MAE | R² |
| :--- | :---: | :---: | :---: |
| Baseline (média) | 131,634 ± 0,158 | 107,582 ± 0,058 | 0,0000 |
| Ridge A (138 colunas) | 110,647 ± 0,175 | 86,601 ± 0,057 | 0,2935 |
| **Ridge B (+ 5 derivadas)** | **110,447 ± 0,179** | **86,383 ± 0,057** | **0,2960** |
| Ridge C (B + SelectKBest, k=50) | 111,284 ± 0,190 | 87,083 ± 0,049 | 0,2853 |
| Ridge D (B com grupos fundidos) | 110,553 ± 0,172 | 86,503 ± 0,050 | 0,2947 |

* O **Ridge B** tem o menor erro e o maior R², nas três dobras, mas a vantagem sobre o A é pequena (cerca de 0,2 ponto). Todos os Ridge erram cerca de 16% menos que o baseline.
* O baseline dá o mesmo resultado nos três conjuntos, porque não usa features.
* **Alpha do Ridge** (0,001 a 10.000.000, nos três conjuntos): de 0,001 a 100 o resultado não muda; só piora a partir de 1 milhão (o modelo fica rígido demais).
* As previsões do Ridge são "achatadas" em direção à média (inclinação da reta de regressão de cerca de 0,29), e ele prevê, em média, 515 pontos para as notas zero.
* Os gráficos de previsão × observado e de erros (todos os participantes do desenvolvimento, previsões fora da dobra) estão em `resultados/figuras/regressao_*.png`. Todos os experimentos estão registrados em `resultados/experimentos.csv`.

### PCA exploratório

O PCA das 138 colunas padronizadas, com todos os 2.153.941 participantes do desenvolvimento, mostra que os dados **não se comprimem bem**: duas componentes explicam 17,5% da variância, três explicam 20,8%, e são necessárias 41 para 80%. A primeira componente liga renda e bens da residência (correlação de −0,457 com a nota). É só visualização: o PCA da modelagem deve ser reajustado dentro de cada dobra.

---

## 🧭 Próximos experimentos e dúvidas

**Ainda não implementados:** regressão linear sem regularização; árvore, KNN, SVR, bagging, pasting, Random Forest, boosting e stacking; PCA dentro da validação cruzada em pelo menos dois regressores; balanceamento; busca de hiperparâmetros nos três melhores; análise de erros por faixa de nota e por grupo; curva de aprendizado; avaliação final no teste (`resultados/previsoes_teste.csv`); e hipóteses de aprofundamento (incluindo comparar a codificação one-hot com números discretos).

**Divisão dos modelos a treinar** (todos nos conjuntos A, B e D, com a mesma tabela de metodologia e resultados do documento):

| Integrante | Modelos |
| :--- | :--- |
| João Igor Vidal de Andrade | Random Forest, Gradient Boosting e Stacking |
| Raquel Albuquerque Quirino | Regressão linear sem regularização e árvore de decisão |
| Amanda Lira | KNN e SVR |
| Sofya Oliveira | Bagging e pasting |

Os três melhores regressores distintos (pelo RMSE médio, sem contar o baseline) seguem para a busca de hiperparâmetros, com pelo menos dez configurações cada.

**Dúvidas que a equipe pretende investigar:** remover, manter ou imputar o "não sei"; efeito das notas zero no treino; se B e D devem ser a base dos próximos modelos; se derivadas e fusão ajudam com modelos de árvore; se UF, dependência e localização da escola e ano de conclusão melhoram o erro; se outro método de seleção ou um `k` maior ajuda; e se os códigos de escola e de município influenciam os resultados.

---

## 💻 Ambiente

1. **Criar e ativar ambiente virtual**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

Os resultados foram gerados com Python 3.14.8, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2 e plotly 7.1.0, em um computador com 10 núcleos e 16 GB de memória. Os estimadores do scikit-learn usados rodam em CPU. Outras versões devem funcionar, mas os números podem variar na última casa decimal. Os 2,15 milhões de participantes do desenvolvimento são lidos em blocos de 100 mil linhas e guardados como matriz esparsa.

---

## 🚀 Ordem de execução

Os scripts devem ser executados a partir da raiz do projeto.

1. **Obtenção dos dados brutos:** baixar e posicionar `MICRODADOS_ENEM_2023.csv` em `dados/raw/` (ver instruções no fim).

2. **Triagem e particionamento 80/20:**
   ```bash
   python src/preparacao_dados.py
   ```
   Conta os 3.933.955 registros, filtra os 2.692.427 com nota de Matemática e salva os identificadores em `dados/splits/` (`desenvolvimento_ids.csv` e `teste_ids.csv`), com `random_state=42`. Se já existirem, só confere que a divisão reproduzida é a mesma.

3. **Diagnóstico da base (opcional):**
   ```bash
   python src/diagnostico_dados.py
   ```
   Ausências por coluna, treineiros, outliers da nota (1,5 × IQR, nada é removido) e verificação do identificador, em `resultados/tabelas/`.

4. **Higienização e codificação (33 features → 138 colunas):**
   ```bash
   python src/higienizacao.py
   ```
   Mantém só as 33 features aprovadas e o alvo (com verificação automática de vazamento) e gera, em `dados/processed/DADOS/`, a base limpa, a base numérica e as versões de desenvolvimento e teste.

5. **Análise exploratória do desenvolvimento:**
   ```bash
   python src/analise_exploratoria.py
   ```
   Distribuição da nota (com as notas zero em destaque) e categorias raras, em `resultados/figuras/` e `resultados/tabelas/`. O teste não é lido.

6. **Dobras da validação cruzada (opcional, já salvas):**
   ```bash
   python src/salvar_dobras.py
   ```
   Guarda a dobra (1, 2 ou 3) de cada participante do desenvolvimento em `dados/splits/dobras_cv.csv`, na ordem do arquivo numérico de desenvolvimento (que é diferente da ordem de `desenvolvimento_ids.csv`; por isso as dobras ficam salvas).

7. **Modelagem preliminar (baseline × Ridge):**
   ```bash
   python src/modelagem_acompanhamento.py
   ```
   Registra em `resultados/experimentos.csv` e `resultados/tabelas/`.

8. **Cada modelo em cada conjunto de features (A, B e D):**
   ```bash
   python src/modelos_por_conjunto.py
   ```
   Os conjuntos e o `Pipeline` de cada um estão em `src/conjuntos_features.py`; a fusão, em `src/fusao_features.py`. Grava `resultados/tabelas/modelos_por_conjunto.csv`.

9. **Varredura exploratória de alpha e gráficos de regressão:**
   ```bash
   python src/varredura_alpha.py
   python src/figuras_regressao_conjuntos.py
   ```
   A varredura (nos três conjuntos) grava `resultados/tabelas/ridge_alpha_cv.csv`; não é a busca oficial de hiperparâmetros. Os gráficos ficam em `resultados/figuras/regressao_ridge_conjunto_*.png`.

10. **Perguntas de pesquisa e PCA exploratório:**
    ```bash
    python src/perguntas_pesquisa.py
    python src/pca_dados_completos.py
    ```
    O primeiro grava as contagens citadas na seção 1.9 do documento. O segundo calcula o PCA com todos os participantes do desenvolvimento e grava as figuras `pca_2d_desenvolvimento.png` e `pca_3d_desenvolvimento.png`, a variância explicada e as cargas.

### Notebooks

Em `notebooks/`: `01_analise_exploratoria*.ipynb` (exploração dos dados), `02_analise_features.ipynb` (features derivadas, redundância, fusão e seleção), `03_limpeza_dados.ipynb` (limpeza e codificação) e `04_modelagem_acompanhamento.ipynb` (baseline e Ridge, gráficos de regressão com todos os dados, cada modelo em cada conjunto de features, e um painel interativo). O painel interativo (`src/ajuste_interativo.py`) é exploração no desenvolvimento e exige `ipywidgets` e um kernel ativo. Os notebooks guardam a exploração; os scripts de `src/` reproduzem o fluxo do acompanhamento.

---

## 🗂️ Estrutura do repositório

```
dados/        raw/ (originais, fora do git), processed/ (preparados, fora do git), splits/ (IDs das partições e dobras)
docs/         documentacao_decisoes.docx e txt/ (diagnósticos e decisões em texto)
notebooks/    exploração e experimentos
src/          preparação, features, modelagem e figuras
resultados/   experimentos.csv, tabelas/ e figuras/
```

---

## 📚 Documentação e referências

* [`docs/documentacao_decisoes.docx`](docs/documentacao_decisoes.docx): documento principal, com as decisões, as tabelas, as figuras, os modelos, o PCA exploratório, os próximos experimentos e as referências.
* [`docs/txt/diagnostico_triagem.txt`](docs/txt/diagnostico_triagem.txt): diagnóstico numérico da base nacional e dos splits.
* [`docs/txt/decisoes_features.txt`](docs/txt/decisoes_features.txt): features escolhidas e removidas, com justificativas.
* [`docs/txt/higienização_e_tratamento_de_vazamentos.txt`](docs/txt/higieniza%C3%A7%C3%A3o_e_tratamento_de_vazamentos.txt): sumário executivo da higienização.

Principais fontes: Inep (Microdados do ENEM 2023 e dicionário), documentação do scikit-learn (Pipeline, Ridge, DummyRegressor, PCA, `SelectKBest`, `FeatureAgglomeration`, `cross_val_predict`), Pedregosa et al. (2011), Spearman (1904), Pearson (1901) e Hoerl; Kennard (1970). A lista completa, com links, está na seção 5 do documento.

---

## 📥 Instruções de obtenção dos dados

Para reproduzir os experimentos, baixe os microdados oficiais:

1. Acesse o portal de dados abertos do Inep, na seção de Microdados do ENEM: [https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).
2. Baixe o arquivo do ano **2023** (`microdados_enem_2023.zip`).
3. Descompacte o pacote e copie o arquivo `MICRODADOS_ENEM_2023.csv` (na pasta `DADOS/` do zip) para:
   ```
   dados/raw/MICRODADOS_ENEM_2023.csv
   ```
   Os scripts também aceitam o zip descompactado sem mover o arquivo (`dados/raw/<pasta_do_zip>/DADOS/MICRODADOS_ENEM_2023.csv`).

> [!CAUTION]
> **ATENÇÃO - ARQUIVO PESADO!**
> O arquivo `MICRODADOS_ENEM_2023.csv` tem cerca de 1,7 GB e **JAMAIS deve ser commitado ou enviado para o GitHub**.
> O `.gitignore` do projeto já ignora a pasta `dados/` (exceto os identificadores das partições e as dobras em `dados/splits/`) e qualquer arquivo `.csv` ou `.zip`. Mantenha essas regras ativas antes de qualquer commit.

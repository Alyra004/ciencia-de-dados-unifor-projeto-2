# Projeto 2 - Ciência de Dados (Previsão da Nota de Matemática - ENEM 2023)

Projeto acadêmico desenvolvido para a disciplina de Ciência de Dados na Universidade de Fortaleza (UNIFOR), focado na análise exploratória, pré-processamento e modelagem preditiva utilizando os microdados do ENEM 2023.

---

## 🎯 Alvo

O objetivo central deste trabalho é construir e avaliar um **modelo de regressão** supervisionado para estimar a nota contínua de **Matemática e suas Tecnologias** dos participantes do ENEM 2023:

* **Variável Alvo:** `NU_NOTA_MT` (Nota da prova de Matemática e suas Tecnologias, escala de 0 a 1000).
* **Tipo de Tarefa:** Regressão supervisionada com base no perfil socioeconômico, demográfico e escolar do candidato (considerando apenas participantes presentes na prova: `TP_PRESENCA_MT = 1`).

---

## 🌐 Fontes

- **Inep (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira)**:
  - Portal de Dados Abertos: [Microdados do ENEM](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)

---

## 📊 Features Utilizadas

O modelo utiliza estritamente atributos conhecidos **antes** da realização do exame, totalizando **37 variáveis preditoras ($X$)**:

1. **Perfil Demográfico e Escolar (10):**
   * `TP_FAIXA_ETARIA`, `TP_SEXO`, `TP_ESTADO_CIVIL`, `TP_COR_RACA`, `TP_NACIONALIDADE`, `TP_ST_CONCLUSAO`, `TP_ANO_CONCLUIU`, `TP_ESCOLA`, `TP_ENSINO`, `IN_TREINEIRO`.
2. **Contexto Pré-Exame (2):**
   * `TP_LINGUA` (opção de língua estrangeira: Inglês ou Espanhol).
   * `SG_UF_PROVA` (UF selecionada para a aplicação da prova).
3. **Questionário Socioeconômico (25):**
   * `Q001` a `Q025` (escolaridade e ocupação dos pais, renda mensal e posse de bens/serviços no domicílio).

> [!IMPORTANT]
> **Prevenção de Data Leakage:**
> Foram completamente excluídas da matriz de entrada $X$ todas as notas de outras áreas (`NU_NOTA_CN`, `CH`, `LC`), notas e competências da redação (`NU_NOTA_REDACAO`, `COMP1` a `COMP5`), códigos de presença (`TP_PRESENCA_*`), situação da redação (`TP_STATUS_REDACAO`), gabaritos e respostas (`TX_*`) e códigos de prova (`CO_PROVA_*`).

---

## 💻 Ambiente

Instruções para configuração do ambiente de desenvolvimento Python:

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

---

## 🚀 Ordem de Execução

1. **Obtenção dos Dados Brutos:**
   * Baixar e posicionar o arquivo `MICRODADOS_ENEM_2023.csv` na pasta `dados/raw/` (ver instruções abaixo).

2. **Triagem e Particionamento (Splits 80/20):**
   ```bash
   python src/preparacao_dados.py
   ```
   * Diagnostica a base de 3.933.955 registros.
   * Filtra os 2.692.427 candidatos com nota em Matemática (`NU_NOTA_MT`).
   * Gera as partições com `random_state=42` e salva os identificadores em `dados/splits/` (`desenvolvimento_ids.csv` e `teste_ids.csv`).

3. **Higienização e Tratamento de Vazamento:**
   ```bash
   python src/higienizacao.py
   ```
   * Filtra estritamente as colunas de leakage.
   * Aplica tratamento categórico sociodemográfico e preenchimento constante no questionário socioeconômico (imune a vazamento em validação cruzada).
   * Salva as bases tratadas em `dados/processed/` (`desenvolvimento_higienizado.csv` e `teste_higienizado.csv`).

4. **Modelagem e Avaliação:**
   * Treinamento dos modelos de regressão, validação cruzada e avaliação final no conjunto de teste isolado.

---

## 📚 Documentação Técnica

Relatórios e diagnósticos detalhados disponíveis na pasta [`docs/`](docs/):
* [`docs/diagnostico_triagem.txt`](docs/diagnostico_triagem.txt): Diagnóstico numérico detalhado da base nacional e dos splits.
* [`docs/relatorio_higienizacao_leakage.md`](docs/relatorio_higienizacao_leakage.md): Inventário completo de Data Leakage, regras de imputação e justificativas metodológicas.
* [`docs/higienização_e_tratamento_de_vazamentos.txt`](docs/higieniza%C3%A7%C3%A3o_e_tratamento_de_vazamentos.txt): Sumário executivo para acompanhamento do projeto.

---

## 📥 Instruções de Obtenção dos Dados

Para reproduzir os experimentos deste projeto, siga os passos abaixo para baixar os microdados oficiais:

1. Acesse o portal de dados abertos do Inep na seção de Microdados do ENEM: [https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).
2. Localize e baixe o arquivo referente ao ano de **2023** (`microdados_enem_2023.zip`).
3. Descompacte o pacote e copie o arquivo de dados principal `MICRODADOS_ENEM_2023.csv` (localizado na pasta `DADOS/` do arquivo zip) para o diretório local do projeto:
   ```
   dados/raw/MICRODADOS_ENEM_2023.csv
   ```
   Os scripts também aceitam o zip descompactado sem mover o arquivo (`dados/raw/<pasta_do_zip>/DADOS/MICRODADOS_ENEM_2023.csv`).

> [!CAUTION]
> **ATENÇÃO - ARQUIVO PESADO!**
> O arquivo `MICRODADOS_ENEM_2023.csv` possui vários gigabytes e **JAMAIS deve ser commitado ou enviado para o GitHub**.
> O arquivo `.gitignore` do projeto já está configurado para ignorar a pasta `dados/` e qualquer arquivo com extensão `.csv` ou `.zip`. Certifique-se de manter essas regras ativas antes de realizar qualquer commit.

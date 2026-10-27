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

*Relação dos atributos socioeconômicos, demográficos e notas escolares selecionados a partir do dicionário de dados do ENEM para a modelagem.*

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

1. Obtenção e extração dos dados brutos em `dados/raw/`.
2. Execução dos scripts/notebooks de pré-processamento e amostragem (`notebooks/` ou `src/`).
3. Geração das bases tratadas (`dados/processed/`) e divisões de treino/teste (`dados/splits/`).
4. Análise exploratória e treinamento dos modelos.
5. Exportação de métricas e gráficos para a pasta `resultados/`.

---

## 📥 Instruções de Obtenção dos Dados

Para reproduzir os experimentos deste projeto, siga os passos abaixo para baixar os microdados oficiais:

1. Acesse o portal de dados abertos do Inep na seção de Microdados do ENEM: [https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).
2. Localize e baixe o arquivo referente ao ano de **2023** (`microdados_enem_2023.zip`).
3. Descompacte o pacote e copie o arquivo de dados principal `MICRODADOS_ENEM_2023.csv` (localizado na pasta `DADOS/` do arquivo zip) para o diretório local do projeto:
   ```
   dados/raw/MICRODADOS_ENEM_2023.csv
   ```

> [!CAUTION]
> **ATENÇÃO - ARQUIVO PESADO!**
> O arquivo `MICRODADOS_ENEM_2023.csv` possui vários gigabytes e **JAMAIS deve ser commitado ou enviado para o GitHub**.
> O arquivo `.gitignore` do projeto já está configurado para ignorar a pasta `dados/` e qualquer arquivo com extensão `.csv` ou `.zip`. Certifique-se de manter essas regras ativas antes de realizar qualquer commit.

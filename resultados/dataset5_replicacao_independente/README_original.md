# Treino independente — Projeto 2 (ENEM 2023)

Este diretório foi criado separadamente e não depende nem altera notebooks, scripts ou resultados preexistentes. A implementação usa somente o PDF `docs/Projeto 2 - Aprendizado de Máquina com Dados do ENEM.docx.pdf` e o CSV indicado pelo usuário.

## Alvo e protocolo

- Alvo escolhido: Matemática (`NU_NOTA_MT`).
- A fonte é lida integralmente; registros sem `NU_NOTA_MT` são contabilizados e retirados apenas da modelagem supervisionada. Notas zero permanecem válidas.
- Separação reproduzível de 80% desenvolvimento / 20% teste (`random_state=42`). IDs são salvos para rastreabilidade, nunca entram em `X`.
- A lista de preditores exclui todas as notas, presenças, respostas, gabaritos, situação da redação e `NU_ANO`.
- Os campos categóricos são codificados dentro dos pipelines dos estimadores. As métricas são calculadas na escala original: MSE, RMSE, MAE e R².

## Execução

```bash
cd treino_projeto2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python treino_enem.py
```

Resultados, partições, previsões e modelos são gravados em `treino_projeto2/resultados/`. Para validar inicialmente a preparação sem treinar, use `--dry-run`.

O script usa a base completa para a comparação treino/teste, mas alguns estimadores (KNN exato e stacking) têm custo quadrático ou múltiplos ajustes internos em milhões de linhas. Eles estão disponíveis sob `--include-expensive` e não são ativados por padrão. A validação cruzada de três dobras, PCA/seleção e busca de hiperparâmetros também são operações separadas e potencialmente longas; este treino de referência não deve ser apresentado como substituto dessas etapas obrigatórias do enunciado.

## Features derivadas

Além dos campos elegíveis de perfil e questionário, são criadas: quantidade de respostas preenchidas no questionário, quantidade de campos de perfil preenchidos, igualdade entre as categorias respondidas para Q001/Q002 e igualdade entre UF da escola e UF da prova. As categorias originais são preservadas sem supor distâncias entre códigos.

## Limites

Município não é usado para controlar a dimensão da codificação categórica; UF é mantida como contexto. O R² pode ser negativo. Métricas do teste descrevem participantes da mesma edição e não demonstram generalização para outros anos, escolas ou municípios.

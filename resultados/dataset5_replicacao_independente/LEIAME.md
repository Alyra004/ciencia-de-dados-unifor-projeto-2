# Dataset 5: solução independente de outra IA (cópia de referência)

Treino feito por outra IA, só com o enunciado e o CSV, sem contexto do nosso trabalho. Mesma partição 80/20 (semente 42): os ids de desenvolvimento e de teste são idênticos aos de `dados/splits/`.

- `treino_enem.py`: o script original. É a **definição do dataset 5** (45 colunas de entrada, one-hot de todas as categóricas, 4 colunas derivadas, modelos de configuração fixa). Foi movido para cá; o caminho `ROOT` dentro dele (`Path(__file__).parents[1]`) assume a pasta antiga, então para rodar é preciso ajustar `ROOT` para a raiz do projeto.
- `metricas_teste.csv`, `melhores_por_metrica.json`, `diagnostico.json`: resultados dela no teste (sem validação cruzada).
- `README_original.md`: o README que ela escreveu.

Não foram guardados: o ambiente virtual, os modelos `.joblib`, os ids (duplicados) e as previsões de teste (61 MB). Resumo dos resultados na seção 8.4 de `docs/documentacao_completa.docx`.

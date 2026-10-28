# Engenharia de features: hipóteses e cuidados

Alvo: `NU_NOTA_MT` (ENEM 2023). Código: [`src/engenharia_features.py`](../src/engenharia_features.py).

Todas as features abaixo são calculadas linha a linha, sem aprender nada com os dados (sem médias, sem codificação supervisionada). Por isso não geram vazamento e podem ser aplicadas antes ou dentro do `Pipeline`. As direções esperadas são **hipóteses a testar**, não resultados. Associação entre perfil e nota não implica causalidade nem mede a capacidade individual.

> Itens marcados **[VERIFICAR]** dependem de conferir a escala no dicionário do ENEM 2023.

## 1. Escolaridade dos responsáveis (`Q001`, `Q002`)

Codificação ordinal: A nunca estudou (0) até G pós-graduação (6). `H` ("não sei") **não é um nível**: vira ausente e é sinalizado à parte.

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `ESC_PAI`, `ESC_MAE` | mapa ordinal de `Q001`/`Q002` | Maior escolaridade dos responsáveis se associa a maior nota. | Distâncias entre níveis não são iguais; o modelo linear assume que são. |
| `ESC_MAX_RESP` | max(`ESC_PAI`, `ESC_MAE`) | O maior nível no domicílio resume o capital educacional melhor que cada um isolado. | Se só um responsável é conhecido, o máximo ignora o outro. |
| `ESC_MEDIA_RESP` | média dos dois | Alternativa mais sensível à diferença entre os responsáveis. | Correlacionada com `ESC_MAX_RESP`; comparar na seleção de atributos. |
| `ESC_DESCONHECE_ALGUM` | 1 se `Q001` ou `Q002` = `H` | Desconhecer a escolaridade carrega informação própria (contexto familiar). | Pode refletir também quem não respondeu com atenção. |

## 2. Renda (`Q005`, `Q006`)

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `RENDA_SM` | ponto médio da faixa de `Q006`, em salários mínimos | Renda maior, nota maior. | É aproximação. `A` = 0. A faixa `Q` (> 20 SM) usa o **piso 20**, o que subestima as rendas mais altas. |
| `RENDA_PER_CAPITA_SM` | `RENDA_SM / Q005` | Renda por morador descreve melhor a condição material que a renda total. | Denominador `Q005` < 1 ou ausente vira ausente. A renda é informada por faixa, então o erro da aproximação se multiplica na divisão. |

## 3. Bens duráveis (`Q010`–`Q018`) [VERIFICAR]

Contagem: A = 0, B = 1, C = 2, D = 3, E = 4 ou mais. O último nível é "N ou mais", então a soma é um **piso**.

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `INDICE_BENS` | soma das contagens | Mais bens indicam maior patrimônio doméstico. | Soma trata um carro e um micro-ondas como equivalentes; é um índice grosseiro. Se um item vier ausente, a soma usa os demais. |
| `BENS_PER_CAPITA` | `INDICE_BENS / Q005` | Bens por morador capturam melhor a posse relativa. | Mesmo cuidado do denominador da renda. |

## 4. Acesso digital (`Q022`, `Q024`, `Q025`)

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `INDICE_DIGITAL` | (celular > 0) + (computador > 0) + (internet = `B`), de 0 a 3 | Acesso a computador e internet favorece o estudo, e a nota acompanha. | Resposta ausente conta como "não tem"; o índice mistura ausência de bem com ausência de resposta. |
| `COMPUTADOR_PER_CAPITA` | contagem de computadores / `Q005` | Computador disponível por morador importa mais que a posse isolada. | Contagem limitada a "4 ou mais". |

## 5. Adensamento e interação

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `MORADORES_POR_QUARTO` | `Q005` / quartos (`Q009` [VERIFICAR]) | Domicílio mais adensado dificulta o estudo. | Quartos = 0 ou ausente vira ausente (denominador inválido). |
| `RENDA_X_ESC_MAX` | `RENDA_SM × ESC_MAX_RESP` | Renda e escolaridade se reforçam: o efeito conjunto supera a soma dos efeitos. | Modelos lineares precisam da interação explícita; árvores já a aprendem sozinhas, então o ganho deve aparecer mais nos lineares. |

## 6. Contexto escolar

| Feature | Fórmula | Hipótese | Cuidado |
|---|---|---|---|
| `ESCOLA_PUBLICA` | 1 se `TP_ESCOLA` = 2 | Escola pública e privada diferem na nota média. | `TP_ESCOLA` = 1 ("não respondeu") fica em 0 junto com a privada; considerar manter a categoria original ao lado. |
| `IDADE_FORA_FAIXA_TIPICA` | 1 se `TP_FAIXA_ETARIA` > 3 | Idade acima da típica sugere defasagem ou EJA. | Treineiros e quem conclui depois não são necessariamente defasados. O corte em 3 vem do dicionário e deve ser conferido. |

## Como comparar (exigência da Seção 6 do enunciado)

Comparar **três representações** nas mesmas dobras (`KFold(3, shuffle=True, random_state=42)`), com o mesmo regressor:

1. **Originais:** só as 37 colunas higienizadas.
2. **Originais + criadas:** as 15 features acima acrescentadas.
3. **Após seleção de atributos:** método a escolher (por exemplo, informação mútua ou importância por árvore), ajustado **dentro** de cada dobra.

Reportar RMSE, MAE e R² (média e desvio entre dobras). Um resultado sem ganho é válido se a hipótese, a execução e a conclusão forem consistentes.

**Aprofundamento (até 0,80 ponto) já viabilizado por estas features:**

- *As interações ajudam?* Comparar com e sem `RENDA_X_ESC_MAX` e as razões per capita.
- *Quais blocos contribuem mais?* Retirar um bloco por vez (renda, escolaridade, bens, digital, localização) e medir a variação do RMSE.

## Visualização por PCA (exploratória)

[`src/pca_visualizacao.py`](../src/pca_visualizacao.py) gera PCA 2D, 3D e variância acumulada sobre uma **amostra de 1% do desenvolvimento** (semente 42; o teste não é lido). A nota só colore os pontos. Por ser diagnóstico, não substitui o experimento de PCA da modelagem, que deve ser reajustado em cada dobra. Ao citar as figuras, informar o tamanho da amostra e que ela é do desenvolvimento.

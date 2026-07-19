# Modelagem inicial governada

## Objetivo

Esta fatia implementa os baselines técnicos do projeto sem atribuir uso clínico ao
resultado. O treinamento parte exclusivamente do snapshot raw validado e do manifesto
correspondente. Nenhum arquivo raw é alterado.

## Normalização canônica

Antes do split, a camada de dados executa somente transformações determinísticas:

- renomeia o alvo raw `0be1dad` para `NObeyesdad`;
- corrige `0rmal_Weight` para `Normal_Weight`;
- normaliza campos binários `0`/`1` para `No`/`Yes`;
- normaliza o valor categórico `0` de `CAEC` e `CALC` para `No`;
- converte os campos numéricos conforme o contrato;
- preserva `id` apenas para auditoria de isolamento das partições.

Schema, domínios, duplicidades, classes, hash e manifesto são novamente validados antes
da modelagem. Uma divergência interrompe o run.

## Split provisório

Na ausência de tempo ou entidade persistente, o contrato inicial usa holdout aleatório
estratificado:

| Partição | Proporção | Uso permitido |
| --- | ---: | --- |
| Treino | 60% | `fit` de preprocessamento e estimadores |
| Validação | 20% | comparação e escolha do baseline |
| Teste | 20% | avaliação única do vencedor |

A seed provisória é `42`. Proporções e seed ficam centralizadas em configuração e
podem ser sobrescritas por CLI ou ambiente. A pipeline comprova que identificadores não
se cruzam, que todas as classes estão presentes e que categorias raras observadas têm
cobertura nas três partições. Se a cobertura falhar, o run para com mensagem acionável.

Essas escolhas ainda precisam de aprovação de negócio/metodologia e não medem
generalização temporal, geográfica ou clínica.

## Preprocessamento

O preprocessamento faz parte da pipeline serializada:

- mediana e padronização para features numéricas;
- moda e one-hot encoding com domínios categóricos explícitos;
- erro para categoria fora do contrato;
- `fit` realizado exclusivamente com treino.

São mantidos dois conjuntos de features:

- `full`: todas as 16 features, sem `id` e sem o alvo;
- `without_anthropometrics`: remove `Height` e `Weight` para auditar circularidade.

## Baselines

O comando compara cinco candidatos fixos, sem busca de hiperparâmetros:

1. regra heurística derivada de IMC;
2. dummy estratificado;
3. regressão logística com features completas;
4. árvore simples com features completas;
5. regressão logística sem altura e peso.

A divisão do intervalo de sobrepeso em níveis I e II na regra de IMC usa o ponto
técnico de `27,5 kg/m²` apenas para reproduzir as sete classes do dataset. Essa divisão
é um baseline de auditoria, não uma recomendação clínica.

O vencedor é escolhido pelo macro F1 de validação. O teste não participa de escolha,
ajuste, threshold ou desempate. O desempate de validação é determinístico pelo nome do
candidato.

## Métricas e artefatos

Cada candidato recebe na validação: macro F1, balanced accuracy, accuracy, log loss,
Brier multiclasse, recall por classe, matriz de confusão e métricas por gênero. Somente
o vencedor recebe o relatório de teste.

Cada run é publicado atomicamente em `artifacts/runs/<run_id>/`:

```text
model.joblib
evaluation.json
manifest.json
```

O manifesto registra dataset, configuração, tamanhos das partições, candidato escolhido
e SHA-256 dos artefatos. O status de MLflow permanece
`not_logged_policy_pending` e nenhuma promoção é solicitada automaticamente.

## Execução

Com o ambiente virtual ativo e o snapshot inicializado:

```bash
obesity-initialize
obesity-train-baselines
```

Sobrescritas explícitas:

```bash
obesity-train-baselines \
  --test-size 0.20 \
  --validation-size 0.20 \
  --random-state 42 \
  --output-root artifacts/runs
```

Os resultados não são um diagnóstico e não estão autorizados para produção. MLflow,
Optuna, calibração pós-treino, gates de promoção e inferência pertencem às próximas
fatias.

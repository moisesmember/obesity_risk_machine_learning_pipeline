# Pipeline de experimentação e modelagem

## 1. Diagnóstico técnico

O projeto resolve uma classificação supervisionada multiclasse com sete estados
corporais ordenados. O snapshot possui 20.758 linhas, baixa cardinalidade categórica,
variáveis contínuas e leve diferença de frequência entre classes. Esse desbalanceamento
não autoriza automaticamente reamostragem nem `class_weight="balanced"`.

O dataset é transversal e parcialmente sintético. Portanto, mede a capacidade de
reconhecer padrões no mesmo processo de geração; não demonstra risco futuro,
causalidade, eficácia clínica ou generalização populacional.

## 2. Riscos e interpretação

- **Vazamento:** informação indisponível na decisão, target ou estatística calculada
  com validação/holdout entra no treinamento. É defeito de correção.
- **Proxy do alvo:** `Weight`, `Height` e `BMI` podem reconstruir aproximadamente a
  regra que originou o estado corporal. Podem estar disponíveis e ainda assim dominar
  o modelo, reduzindo a interpretação comportamental.
- **Feature legítima:** uma medida comprovadamente disponível na inferência pode ser
  usada, desde que o objetivo real aceite essa dependência e ela seja monitorada.

`Gender` também pode atuar como atalho devido às associações observadas no snapshot.
As métricas por gênero e a ablação B avaliam essa dependência sem atribuir causalidade.

## 3. Estratégia de validação

O split é criado antes de qualquer `fit`:

| Partição | Proporção | Uso |
| --- | ---: | --- |
| Desenvolvimento | 80% | cinco folds estratificados para comparação, features e Optuna |
| Holdout final | 20% | uma avaliação confirmatória da configuração vencedora |

O desenvolvimento usa `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)`. Cada fold recebe um clone completo da pipeline; imputação,
padronização, encoding e feature engineering são ajustados apenas no treino do fold.

## 4. Experimentos de features

O catálogo está em `features/experiments.py`, enquanto a seleção executada fica em
`configs/experiments.json`.

| Experimento | Features/representação | Objetivo |
| --- | --- | --- |
| A_full | 16 preditoras, Age contínua | referência completa |
| A_full_ordinal | A, com CAEC/CALC ordinais | nominal versus ordinal |
| B_without_gender | A sem Gender | dependência do atalho de gênero |
| C_without_weight | A sem Weight, mantendo Height | efeito isolado do peso |
| C_without_weight_height | A sem Weight/Height | remoção dos proxies corporais |
| D_behavioral | hábitos, atividade, água, tecnologia, transporte e histórico | estimativa comportamental |
| E_body_bmi | Weight, Height e BMI | quanto a regra corporal explica |
| F_age_continuous | A com Age float | representação principal |
| F_age_completed | A com `floor(Age)` | entrada em anos completos |
| F_age_grouped | A com faixas | entrada categórica interpretável |

`Age`, `Age_completed` e `Age_group` nunca coexistem no mesmo experimento. BMI é
calculado dentro da pipeline por `Weight / Height²`; `id` e `NObeyesdad` nunca entram
nas features.

## 5. Validação dos dados

A camada de schema equivalente a frameworks dedicados valida:

- colunas obrigatórias e inesperadas;
- tipos, nulos, infinitos e unicidade de `id`;
- sete classes e tolerância configurável de proporção;
- domínios categóricos;
- limites numéricos centralizados em `configs/modeling.json`;
- manifesto, tamanho e SHA-256 do snapshot;
- perfil de distribuição e alertas PSI para lotes de inferência.

O treino rejeita categorias fora do contrato. A inferência aceita categorias nominais
novas para modelos configurados com `handle_unknown="ignore"`, mantendo o evento de
drift observável.

## 6. Preprocessamento

- valores numéricos permanecem contínuos no experimento principal;
- regressão logística recebe mediana e padronização;
- árvores recebem mediana sem padronização obrigatória;
- categorias nominais usam one-hot com categorias desconhecidas ignoradas;
- CAEC/CALC possuem experimento ordinal explícito;
- CatBoost recebe `DataFrame` e nomes categóricos diretamente, sem one-hot manual.

Nenhuma etapa usa target encoding ou balanceamento nesta versão.

## 7. Modelos progressivos

O catálogo implementa DummyClassifier, LogisticRegression, ExtraTrees,
RandomForest, HistGradientBoosting, CatBoost, LightGBM e XGBoost. Backends externos são
carregados sob demanda. Ausência de pacote vira `unavailable`; falha de um backend
instalado vira `failed`, sem invalidar candidatos independentes.

Instale o catálogo completo com:

```bash
python -m pip install -r requirements-modeling.txt
```

## 8. Métricas e seleção

Cada fold registra macro F1, weighted F1, accuracy, balanced accuracy, log loss,
Brier multiclasse, precision/recall/F1 por classe, matriz de confusão, erro absoluto
ordinal, Quadratic Weighted Kappa e métricas por gênero. O relatório agrega média e
desvio-padrão.

O ranking usa: maior macro F1 médio, menor desvio do macro F1, menor erro ordinal,
maior Kappa quadrático, menor tempo e nome determinístico. Accuracy nunca é o único
critério.

`leaderboard.csv` contém experimento, features, idade, modelo, macro F1 médio/desvio,
accuracy, log loss, erro ordinal, tempo e observação sobre proxies.

## 9. Optuna

Optuna é desabilitado por padrão (`0` trials) e só recebe o candidato selecionado pela
comparação inicial. A objective maximiza macro F1 médio nos mesmos folds de
desenvolvimento e registra estabilidade, erro ordinal e tempo. O holdout nunca é
passado ao estudo.

```bash
obesity-run-experiments --optuna-trials 20
```

## 10. MLflow

Quando habilitado, um run pai registra identidade do dataset, configuração do split,
quantidade de linhas por partição, hiperparâmetros finais do vencedor, duração das
etapas, métricas do holdout e artefatos governados. Cada candidato recebe um run filho
identificável por modelo, feature set e representação de idade, com parâmetros, médias,
desvios, métricas por classe e histórico dos folds usando `step`. Quando a otimização
está ativa, cada trial do Optuna também é um run filho com estado, parâmetros e
métricas, além do resumo consolidado no run pai.

Falha do tracking é registrada como `failed_optional` e não apaga o resultado local
completo. O tracking é manual para preservar a hierarquia do experimento e impedir que
autologging crie runs duplicados.

```bash
obesity-run-experiments --enable-mlflow
```

O MLflow recebe modelo, avaliação, leaderboard, explicabilidade, perfil de
distribuição, resumo do Optuna, ambiente, eventos de treinamento e manifesto. O arquivo
`predictions.csv` permanece exclusivamente local porque contém `id`, classe real e
resultado por registro. O tracking não recebe dados raw, credenciais ou exemplos
sensíveis.

### Logs operacionais e trilha de auditoria

O CLI escreve no console eventos estruturados de carga, contrato, split, catálogo,
validação de candidatos, cada fold, seleção, Optuna, fit final, holdout,
explicabilidade, artefatos, MLflow e publicação. Use `--log-level DEBUG`, `INFO`,
`WARNING` ou `ERROR` nos entry points de treinamento.

Os mesmos eventos ficam em `training_events.jsonl`, um JSON por linha com timestamp
UTC, `run_id`, etapa, estado, duração e métricas seguras. Esse arquivo entra no
manifesto com SHA-256 e também é enviado ao run pai do MLflow.

## 11. Explicabilidade

O vencedor recebe permutation importance por feature raw em amostra limitada. O
relatório destaca Weight, Height, BMI, Gender e Age quando relevantes. SHAP é detectado
como integração opcional; pipelines genéricas incompatíveis são marcadas explicitamente
e não recebem explicações fabricadas.

## 12. Artefatos e inferência

Cada execução é publicada atomicamente em `artifacts/runs/<run_id>/`:

```text
model.joblib
evaluation.json
leaderboard.csv
predictions.csv
explainability.json
explainability.png
distribution_profile.json
optuna.json
environment.json
training_events.jsonl
manifest.json
```

`predictions.csv` preserva `id`, classe real no holdout, classe prevista e sete
probabilidades. O manifesto contém hashes e o contrato de inferência. A serialização é
recarregada e comparada com o modelo em memória antes da publicação.

Inferência batch:

```bash
obesity-predict \
  --run-directory artifacts/runs/<run_id> \
  --input data/processed/inference.csv \
  --output data/predictions/predictions.csv
```

## 13. Estrutura modular

```text
configs/                 defaults e plano de experimentos
src/obesity_risk_pipeline/
  config/                configuração tipada
  data/                  contrato, validação, split e drift
  features/              engenharia, ablações e preprocessing
  models/                catálogo, métricas, CV, Optuna, tracking e explicabilidade
  pipelines/             orquestração de treino
  inference/             carregamento e predição batch
tests/unit/              contratos, leakage, paridade e reprodutibilidade
```

## 14. Critérios para um modelo final

O vencedor estatístico ainda não está promovido. A decisão humana deve considerar
macro F1, estabilidade, recall das sete classes, erro ordinal, log loss, dependência de
proxies/Gender, interpretação, latência e coerência com a entrada real. Sem consumidor,
custo de erro e contrato de produção aprovados, o status permanece
`promotion_status=not_requested`.

## 15. Execução recomendada

Automação completa no Windows/PowerShell:

```powershell
# smoke test: venv + dependências + ingestão idempotente + treino
.\scripts\run_training_pipeline.ps1

# catálogo completo com quality gate e tracking local
.\scripts\run_training_pipeline.ps1 `
  -Mode full `
  -RunTests `
  -StartInfrastructure `
  -EnableMlflow `
  -OptunaTrials 20
```

O script usa diretamente o Python do `.venv`, valida cada código de saída e interrompe
na primeira falha. `-SkipInstall` torna execuções subsequentes mais rápidas. A etapa de
ingestão continua idempotente: snapshot existente só é reutilizado após validação.

Em ambientes já preparados, use diretamente o entry point multiplataforma:

```bash
obesity-training-pipeline --mode quick
obesity-training-pipeline --mode full --config configs/experiments.json
```

Execução manual equivalente:

```bash
obesity-initialize
obesity-train-baselines
obesity-run-experiments --config configs/experiments.json
```

O primeiro treino é um smoke test rápido. O segundo executa o catálogo completo e pode
demandar vários minutos. Os resultados servem a aprendizado, benchmarking e
demonstração de MLOps, nunca diagnóstico clínico.

## Referências técnicas

- [Obesity Risk Dataset — Kaggle](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)
- [StratifiedKFold — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [CatBoostClassifier e categorias nativas](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
- [MLflow Python API](https://mlflow.org/docs/latest/api_reference/python_api/index.html)

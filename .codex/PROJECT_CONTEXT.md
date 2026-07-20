# Contexto da pipeline de risco de obesidade

## Identidade e objetivo

- Projeto: `obesity_risk_machine_learning_pipeline`.
- Problema de negócio: classificar o estado de peso associado a um perfil individual
  usando medidas corporais, hábitos alimentares e aspectos do estilo de vida.
- Tipo de ML: classificação supervisionada multiclasse com sete classes.
- Unidade de predição: um perfil individual representado por uma linha do dataset.
- Target e significado: coluna raw `0be1dad`, normalizada para `NObeyesdad`; representa
  peso insuficiente, peso normal, dois níveis de sobrepeso ou três tipos de obesidade.
- Momento em que a predição ocorre: fotografia transversal no momento do preenchimento
  das informações; não existe janela futura nem coluna temporal.
- Consumidor da predição: decisão de negócio em aberto; uso atual educacional,
  experimental e analítico.
- Ação tomada a partir do resultado: em aberto; nenhuma ação clínica está autorizada.
- Custos de erro ou restrições operacionais: em aberto.

## Contrato dos dados

- Fonte: dataset público
  `jpkochar/obesity-risk-dataset`, hospedado no Kaggle.
- Formato e localização raw: `obesity_level.csv`, publicado de forma imutável em
  `data/raw/obesity_risk_dataset/<sha256>/` pela ingestão.
- Chave: `id`, único e sequencial no snapshot conhecido; não é uma feature.
- Coluna temporal e timezone: não existem.
- Como o label é obtido: a fonte fornece a classe pronta; sua regra completa de geração
  não está documentada na versão Kaggle.
- Colunas obrigatórias: `id`, `Gender`, `Age`, `Height`, `Weight`,
  `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`,
  `SCC`, `FAF`, `TUE`, `CALC`, `MTRANS` e `0be1dad`.
- Política de duplicidade: falhar se `id` repetir ou se houver registro duplicado após
  desconsiderar `id` na validação de dados que antecede o treino.
- Política para target ausente/ambíguo: falhar; nunca converter para uma classe.
- Dados sensíveis: gênero, idade, medidas corporais e hábitos relacionados à saúde
  exigem acesso controlado; não registrar linhas ou exemplos reais no tracking.
- Snapshot conhecido: 20.758 linhas, 18 colunas, 2.444.336 bytes, SHA-256
  `04549179841220E7537EE9065FAC9CF9446C6368133882B7199A5618EA541EE6`.
- Identificação/versionamento: slug da fonte, data UTC da ingestão, hash SHA-256,
  tamanho, schema, contagem de linhas e distribuição do alvo no manifesto.

## Disponibilidade point-in-time

- Features disponíveis: a fonte pressupõe que todas as 16 features são informadas na
  mesma fotografia do perfil; a aplicação real ainda precisa confirmar essa hipótese.
- Features pós-evento: não identificadas porque não existe evento temporal.
- Agregações históricas: não aplicável ao snapshot atual.
- Entidades que não podem cruzar partições: não identificáveis; `id` representa linha,
  não entidade persistente.
- Riscos de leakage: `Weight` e `Height` permitem derivar IMC e podem reproduzir a regra
  do alvo; `Gender` possui associação quase determinística com algumas classes; dados
  sintéticos podem conter registros muito semelhantes entre partições.

## Split e validação

- Estratégia provisória: holdout estratificado, pois não há tempo ou grupo disponível.
- Justificativa: estima apenas generalização para registros intercambiáveis do mesmo
  processo de geração; não demonstra generalização temporal, geográfica ou clínica.
- Proporções governadas: 80% desenvolvimento e 20% holdout final, com seed `42`.
- Comparação e seleção usam `StratifiedKFold` com cinco folds, shuffle e seed fixa
  exclusivamente dentro do desenvolvimento.
- Out-of-time/backtesting: não aplicável sem uma fonte temporal.
- Invariante: separar holdout antes de qualquer fit, seleção, imputação, otimização ou
  amostragem; o holdout recebe somente a configuração final.

## Modelagem e avaliação

- Experimentos obrigatórios: completo, sem gênero, sem peso, sem peso/altura,
  comportamental, corporal com BMI e três representações exclusivas de idade.
- Famílias permitidas: Dummy, regressão logística, ExtraTrees, RandomForest,
  HistGradientBoosting, CatBoost, LightGBM e XGBoost.
- Métrica principal: macro F1 médio da validação cruzada.
- Confirmação: desvio entre folds, balanced accuracy, weighted F1, precision/recall/F1
  por classe, matriz de confusão, log loss, Brier, erro ordinal, Kappa quadrático e
  métricas por gênero.
- Threshold: em aberto; argmax é apenas comportamento técnico inicial.
- Restrições de latência, tamanho e explicabilidade: em aberto.

## MLflow e rastreamento de experimentos

- MLflow permanece opcional para experimentos que não serão promovidos; quando
  desabilitado ou indisponível, o resultado local permanece completo. A política de
  promoção `experimental-v1` exige que o run candidato esteja registrado no MLflow.
- Tracking local previsto: `http://localhost:5000`.
- Backend previsto: PostgreSQL; artefatos previstos: MinIO compatível com S3.
- Convenção: um run pai por execução completa; runs filhos somente para trials ou
  challengers quando necessário.
- Tags mínimas: versão/hash do dataset, versão do código e feature set.
- Não registrar dados raw, segredos ou exemplos sensíveis.
- O Model Registry possui adaptador governado: somente uma decisão aprovada, com
  identidade do aprovador e ticket, pode registrar uma versão e atribuir um alias.
- Os aliases default são `candidate` e `champion`; a política versionada pode
  restringi-los. Autologging e retenção permanecem em aberto.

## Optuna e busca de hiperparâmetros

- Optuna é opcional, desabilitado por padrão e executado somente após a comparação
  inicial, sobre o candidato selecionado.
- Objective: macro F1 médio nos mesmos folds de desenvolvimento; o estudo também
  registra estabilidade, erro ordinal e tempo e nunca recebe o holdout.
- O número de trials é limitado por `OBESITY_OPTUNA_TRIALS`; zero preserva o caminho
  completo sem otimização.

## AutoML e challengers

- AutoGluon, H2O e FLAML são benchmarks opcionais declarados em
  `requirements-benchmarks.txt`.
- Métrica, split interno, orçamento, recursos, ensembles e portabilidade: em aberto.
- Falha de benchmark opcional não pode invalidar o treinamento principal.
- Nenhum backend pode promover um modelo automaticamente.

## Governança e promoção

- Artefatos iniciais obrigatórios da ingestão: CSV raw imutável e `manifest.json` com
  proveniência, hash, tamanho, schema, contagens e timestamp UTC.
- Object storage previsto: MinIO; banco de metadados previsto: PostgreSQL.
- Snapshots de dados no MinIO: bucket `obesity-risk-datasets`, prefixo
  `datasets/obesity_risk_dataset/<sha256>/`, com CSV e manifesto imutáveis.
- Artefatos do MLflow no MinIO: bucket `obesity-risk-mlflow`, prefixo `artifacts/`;
  ambos os buckets são criados de forma idempotente pelo bootstrap do Compose.
- A política `experimental-v1` está versionada em `configs/promotion.json`. Ela exige
  no holdout macro F1 e balanced accuracy >= `0.82`, recall de cada classe >= `0.68`
  e erro ordinal <= `0.20`; exige também desvio do macro F1 entre folds <= `0.02` e
  diferença de macro F1 entre gêneros <= `0.05`.
- Esses limites são gates para uso experimental no snapshot conhecido, derivados com
  margem sobre quatro runs reproduzíveis. Não são critérios clínicos nem baseline de
  produção externa e precisam ser revistos quando dataset, feature set, população ou
  uso de negócio mudar.
- A política exige tracking MLflow, nenhum candidato com falha e aprovação humana com
  ticket. Backends opcionais indisponíveis são tolerados, mas o candidato selecionado
  precisa estar íntegro e carregável.
- Promoção e rollback são operações separadas do treino. Ambas exigem gates aprovados,
  aprovador e ticket e produzem relatórios locais imutáveis; rollback aponta uma nova
  decisão `champion` para um run histórico aprovado, sem sobrescrever histórico.
- Não há promoção automática autorizada.
- Cada execução publica atomicamente modelo, avaliação, leaderboard, previsões,
  explicabilidade, perfil de distribuição, estudo Optuna, ambiente, eventos JSONL de
  treinamento e manifesto com hashes.
- O MLflow recebe parâmetros, durações, métricas de holdout, runs filhos dos
  candidatos/folds e artefatos governados. `predictions.csv` permanece local por
  conter identificadores e resultados por registro.
- O GitHub Actions executa o check `quality-gate` em pull requests e pushes para
  `main`; a proteção da branch deve exigir esse check antes do merge.
- A imagem da API é construída pelo quality gate. Destino, registry de imagens,
  credenciais e rollout permanecem em aberto; o workflow não publica nem promove
  artefatos.

## Inferência e operação

- Modos implementados: batch CSV local ou MinIO por URI `s3://` via
  `obesity-predict`, e API FastAPI via `obesity-serve` ou profile `serving` do Compose.
- A API inicia de forma fail-closed e só fica ready com run e relatório de promoção
  íntegros, aprovados e compatíveis. Schema inválido, target/campos extras e valores
  fora dos limites governados são rejeitados. Categorias novas são toleradas pelo
  preprocessador e aparecem no cálculo de drift, conforme o contrato executável atual.
- Telemetria implementada: run do modelo, volume, classes previstas, confiança média,
  alertas de drift e duração, sem registrar perfis individuais. Frequência, SLA,
  autenticação, retenção, alertas e feedback com labels permanecem em aberto.

## Stack e estrutura

- Python: compatível com as bibliotecas versionadas em `requirements.txt`; o projeto
  empacotado exige Python 3.10 ou superior.
- Estrutura: pacote `src/obesity_risk_pipeline`, testes em `tests`, comandos em
  `scripts` ou entry points e documentação em `docs`.
- Configuração: defaults seguros e variáveis de ambiente; credenciais somente via
  ambiente ou arquivo oficial do Kaggle fora do repositório.
- Ambientes documentados: Windows/PowerShell e Linux/macOS; benchmarks podem exigir
  Linux/WSL.

## Comandos verificáveis

```bash
# criar ambiente (Windows)
py -m venv .venv

# instalar dependências e o pacote local
python -m pip install -r requirements.txt
python -m pip install -e .

# inicialização idempotente: valida o snapshot existente ou importa do Kaggle
obesity-initialize

# automação Windows: venv, dependências, ingestão e treino rápido
.\scripts\run_training_pipeline.ps1

# orquestração multiplataforma em ambiente já instalado
obesity-training-pipeline --mode quick

# testes
python -m pytest

# mesmo quality gate executado pelo GitHub Actions
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e .
python -m compileall -q src tests
docker compose config --quiet
python -m pytest -q tests/unit

# infraestrutura local
docker compose config
docker compose up --build -d

# exploração conectada ao MinIO
docker compose up -d minio
python -m jupyter lab notebooks/01_data_exploration_minio.ipynb

# treino governado dos baselines no snapshot canônico
obesity-train-baselines

# catálogo completo de ablações e modelos
python -m pip install -r requirements-modeling.txt
obesity-run-experiments --config configs/experiments.json

# inferência batch com um run governado
obesity-predict --run-directory artifacts/runs/<run_id> --input <input.csv> --output <output.csv>

# avaliar gates; aprovação e --register são fornecidos somente após revisão humana
obesity-promote --run-directory artifacts/runs/<run_id> --policy configs/promotion.json

# API local (exige OBESITY_MODEL_RUN_DIRECTORY e OBESITY_PROMOTION_REPORT)
obesity-serve

# imagem/API pelo Docker Compose
docker compose --profile serving up --build -d inference-api
```

API de inferência e gates de promoção possuem testes verificáveis. Lint, type-check e
migrations ainda não estão configurados. MLflow é opcional no treino, mas pode ser
obrigatório pela política de promoção.

## Critérios de aceite iniciais

- A ingestão baixa por um adaptador Kaggle isolado e nunca publica download parcial.
- A inicialização reutiliza um snapshot existente somente após validar sua integridade;
  quando ele não existe, executa a ingestão automaticamente.
- O schema, o target, a unicidade do `id` e o hash esperado são validados antes da
  publicação.
- O raw é versionado por SHA-256, não é sobrescrito e a repetição é idempotente.
- O manifesto preserva proveniência sem credenciais ou registros individuais.
- Testes não dependem de internet nem de credenciais reais.
- README e documentação de negócio refletem o comando e o layout publicados.

## Decisões em aberto

- Consumidor, ação e custo dos erros.
- Uso de classificação atual ou risco futuro.
- População-alvo e fonte externa de validação.
- Disponibilidade de altura e peso na decisão real.
- Baseline externo e critérios clínicos/operacionais para substituir a política
  experimental quando houver um uso real definido.
- Política de disponibilidade e retenção do MLflow.
- Identidade dos aprovadores e sistema oficial de tickets.
- Autenticação/autorização da API, SLA, retenção, alertas e feedback com labels.
- Destino de deploy, registry da imagem, credenciais e estratégia de rollout do CD.

# Promoção, Registry e API de inferência

## Objetivo e limite de uso

Este projeto possui um caminho técnico para avaliar, registrar e servir um modelo sem
misturar treinamento com promoção. A API continua destinada a uso experimental: o
dataset não sustenta diagnóstico médico, causalidade nem generalização clínica.

O fluxo implementado é:

`run imutável -> gates -> aprovação humana -> MLflow Registry -> API fail-closed`

Treinar nunca promove um modelo. O comando de promoção lê os artefatos já publicados,
gera uma decisão auditável e somente altera o Registry quando todos os gates e a
aprovação humana estiverem presentes.

## Pré-requisitos

Instale as dependências de modelagem e suba MLflow, PostgreSQL e MinIO:

```powershell
python -m pip install -r requirements-modeling.txt
python -m pip install -e .
docker compose up --build -d postgres minio minio-init mlflow
```

O treino que será candidato à promoção precisa usar MLflow:

```powershell
$env:OBESITY_MLFLOW_ENABLED = "true"
obesity-run-experiments --config configs/experiments.json
```

Anote o diretório `artifacts/runs/<run_id>` exibido ao final.

## Política de promoção

O contrato está em
[`configs/promotion.schema.json`](../configs/promotion.schema.json), e a política
inicial está versionada em
[`configs/promotion.json`](../configs/promotion.json). Ela foi calibrada para uso
experimental a partir dos quatro runs reproduzíveis existentes:

| Gate | Limite v1 | Faixa observada nos runs completos |
| --- | ---: | ---: |
| Macro F1 no holdout | >= 0,82 | 0,854–0,859 |
| Balanced accuracy no holdout | >= 0,82 | 0,855–0,860 |
| Recall de cada classe | >= 0,68 | mínimo de 0,704–0,715 |
| Erro ordinal no holdout | <= 0,20 | 0,141–0,147 |
| Desvio do macro F1 entre folds | <= 0,02 | 0,0051 |
| Diferença de macro F1 entre gêneros | <= 0,05 | 0,0004–0,0019 |

As folgas evitam acoplar a aprovação a uma única execução. MLflow, ausência de
candidatos falhos e aprovação humana são obrigatórios; challengers opcionais
indisponíveis não bloqueiam. Essa política não autoriza uso clínico e deve ganhar nova
versão quando houver baseline externo, custo dos erros ou mudança de dados/features.

## Gates executados

`obesity-promote` falha de forma fechada quando falta métrica ou artefato. A decisão
verifica:

- macro F1, balanced accuracy e erro ordinal do holdout;
- recall de todas as sete classes do target;
- estabilidade do macro F1 entre folds;
- diferença de macro F1 entre os grupos de gênero observados;
- existência, tamanho e SHA-256 dos artefatos obrigatórios do run;
- paridade declarada entre o modelo em memória e o serializado;
- tracking MLflow, candidatos falhos/indisponíveis conforme a política;
- nome do aprovador e ticket de mudança quando a aprovação humana é obrigatória.

Primeiro execute apenas a avaliação técnica:

```powershell
obesity-promote `
  --run-directory artifacts/runs/<run_id> `
  --policy configs/promotion.json
```

Os códigos de saída são `0` para aprovado, `2` para tecnicamente aprovado aguardando
aprovação, `3` para rejeitado e `1` para erro de contrato/operação.

Depois da revisão humana, registre o modelo com uma identidade rastreável:

```powershell
obesity-promote `
  --run-directory artifacts/runs/<run_id> `
  --policy configs/promotion.json `
  --approved-by "responsavel@empresa.com" `
  --approval-ticket "CHANGE-1234" `
  --register `
  --alias candidate
```

O modelo é logado no formato seguro `skops`, com allowlist estática apenas para os
transformers internos governados, código do pacote, signature e exemplo sintético. Em
seguida, é recarregado pelo MLflow e testado quanto à paridade antes do registro. O
relatório e seu manifesto são publicados em
`artifacts/promotions/<run_id>/<decision_id>/`. Repetir exatamente a mesma decisão
reutiliza esse relatório.

Para usar o alias `champion`, faça uma nova decisão com o ticket que autoriza a troca.
Para rollback, execute a mesma promoção apontando para um run histórico aprovado e um
novo ticket, usando `--alias champion`. O procedimento cria uma nova versão rastreável
do artefato histórico e não sobrescreve o run nem o relatório anterior.

## Executar a API

Instale somente o conjunto de serving quando o ambiente não precisar treinar. Esse
arquivo fixa as versões e inclui todos os backends que podem vencer o catálogo para
garantir que o artefato serializado seja carregável. O XGBoost usa sua distribuição
CPU-only, pois a API não configura inferência em GPU:

```powershell
python -m pip install -r requirements-serving.txt
python -m pip install -e .
```

Para executar localmente sem container, configure os caminhos do run e do relatório:

```powershell
$env:OBESITY_MODEL_RUN_DIRECTORY = "artifacts/runs/<run_id>"
$env:OBESITY_PROMOTION_REPORT = "artifacts/promotions/<run_id>/<decision_id>/promotion_report.json"
$env:OBESITY_REQUIRE_PROMOTION = "true"
obesity-serve
```

Com Docker Compose, monte ambos como somente leitura:

```powershell
$env:OBESITY_SERVING_RUN_DIRECTORY = (Resolve-Path "artifacts/runs/<run_id>")
$env:OBESITY_SERVING_PROMOTION_DIRECTORY = (Resolve-Path "artifacts/promotions/<run_id>/<decision_id>")
docker compose --profile serving up --build -d inference-api
```

Verifique a prontidão e consulte o contrato OpenAPI:

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
Start-Process http://localhost:8000/docs
```

`/health/live` confirma apenas o processo. `/health/ready` retorna `503` se o modelo,
manifesto ou relatório aprovado estiver ausente, corrompido ou pertencer a outro run.
A rota `POST /v1/predict` rejeita target, campos extras, valores fora dos limites
governados e lotes acima do limite configurado. Categorias novas seguem a política do
preprocessador e contribuem para a telemetria de drift.

A resposta identifica o `model_run_id` e inclui somente telemetria agregada: volume,
contagem por classe prevista, confiança média máxima, alertas de drift e duração. A API
não registra perfis individuais nem payloads de entrada.

## Controles operacionais ainda dependentes do ambiente

O repositório entrega uma imagem reproduzível e uma composição local, mas não escolhe
por conta própria Kubernetes, ECS, Cloud Run ou outro destino. Antes de tráfego real,
a organização ainda deve definir autenticação/autorização, TLS, secret manager, SLO,
autoscaling, retenção de logs, alertas, validação externa, monitoramento com labels e o
processo formal de incidentes. O workflow de CI constrói a imagem para provar que ela
é empacotável, porém não publica nem faz deploy sem esse destino e suas credenciais.

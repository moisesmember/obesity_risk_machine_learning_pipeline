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
- Proporções, cross-validation, política de amostragem e seed: decisões em aberto para
  a fatia de modelagem.
- Out-of-time/backtesting: não aplicável sem uma fonte temporal.
- Invariante: separar teste antes de qualquer fit, seleção, imputação ou amostragem.

## Modelagem e avaliação

- Baselines obrigatórios: regra derivada de IMC; dummy estratificado; modelo linear
  multiclasse; árvore simples; challenger sem peso/IMC.
- Famílias permitidas: ainda não governadas; Scikit-learn é dependência principal.
- Métrica principal provisória: macro F1, sujeita à confirmação do custo de negócio.
- Confirmação: balanced accuracy, recall por classe, matriz de confusão, log loss,
  calibração e métricas por gênero quando houver amostra suficiente.
- Threshold: em aberto; argmax é apenas comportamento técnico inicial.
- Restrições de latência, tamanho e explicabilidade: em aberto.

## MLflow e rastreamento de experimentos

- MLflow está previsto, mas a política de obrigatoriedade e falha está em aberto.
- Tracking local previsto: `http://localhost:5000`.
- Backend previsto: PostgreSQL; artefatos previstos: MinIO compatível com S3.
- Convenção: um run pai por execução completa; runs filhos somente para trials ou
  challengers quando necessário.
- Tags mínimas: versão/hash do dataset, versão do código e feature set.
- Não registrar dados raw, segredos ou exemplos sensíveis.
- Model Registry, autologging, retenção, aliases e responsável por promoção: em aberto.

## Optuna e busca de hiperparâmetros

- Optuna é dependência disponível, mas não faz parte da fatia de ingestão.
- Objective, folds, sampler, pruner, seed, espaços, orçamentos, storage, retomada e
  refit: decisões em aberto.
- Caminho sem Optuna será obrigatório.

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
- Gates, baseline de produção, registry, rollback, retenção e responsável por aprovação
  humana: em aberto.
- Não há promoção automática autorizada.
- O GitHub Actions executa o check `quality-gate` em pull requests e pushes para
  `main`; a proteção da branch deve exigir esse check antes do merge.
- O destino e a estratégia de deploy permanecem em aberto; o workflow atual não
  publica nem promove artefatos.

## Inferência e operação

- Modo, contrato de entrada/saída, frequência, SLA e feedback: em aberto.
- Entradas com schema inválido ou artefato incompatível deverão ser rejeitadas com erro
  acionável.
- Monitoramento futuro: schema, drift, classes previstas, confiança, versão e métricas
  por grupo quando labels estiverem disponíveis.

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
```

Treino, inferência, lint, type-check, migrations e gates de promoção ainda não possuem
implementação verificável.

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
- Split, proporções, seed e cross-validation da etapa de modelagem.
- Métricas e gates numéricos de promoção.
- Política de disponibilidade do MLflow.
- Contrato de inferência, SLA, monitoramento e aprovação humana.
- Destino de deploy, artefato implantável, credenciais, estratégia de promoção e
  rollback do CD.

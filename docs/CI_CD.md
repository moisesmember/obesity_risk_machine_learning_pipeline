# Integração contínua e entrega

## Objetivo

O workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) funciona como o
quality gate do repositório. Ele é executado:

- em pull requests direcionados à branch `main`;
- após cada push recebido pela branch `main`;
- manualmente, pela opção **Run workflow** do GitHub Actions.

O job possui o nome estável `quality-gate`, permissões somente de leitura, timeout de
15 minutos e cancelamento de execuções obsoletas da mesma referência.

## Validações executadas

O gate usa Python 3.10, a menor versão suportada pelo pacote, e executa:

1. instalação das dependências mínimas de CI e do pacote local em modo editável;
2. verificação de consistência das dependências instaladas;
3. compilação estática dos módulos em `src/` e `tests/`;
4. validação declarativa do `docker-compose.yml`, sem iniciar containers;
5. construção da imagem Docker da API de inferência;
6. testes unitários em `tests/unit`.

Os testes unitários não acessam Kaggle, MinIO ou internet. Os adaptadores externos são
simulados e o notebook é verificado estaticamente para garantir JSON válido, código
Python compilável e ausência de resultados ou contadores de execução versionados.

## Reprodução local

Crie e ative o ambiente virtual e execute os mesmos comandos do job:

```bash
python -m pip install --upgrade pip
python -m pip install --requirement requirements-ci.txt
python -m pip install --no-deps --editable .
python -m pip check
python -m compileall -q src tests
docker compose config --quiet
docker build --file docker/api/Dockerfile --tag obesity-risk/inference-api:ci .
python -m pytest -q tests/unit
```

O arquivo `requirements-ci.txt` é intencionalmente mínimo. A instalação completa para
notebooks, rastreamento e modelagem continua definida em `requirements.txt`.

## Proteção da branch `main`

Um workflow disparado por `push` detecta problemas depois que o código já entrou na
branch. Para impedir a entrada de mudanças com falha, configure no GitHub uma regra de
proteção (ou ruleset) para `main` com estas condições:

1. exigir pull request antes do merge;
2. exigir que os status checks estejam aprovados;
3. selecionar o check obrigatório `quality-gate`;
4. exigir que a branch esteja atualizada antes do merge;
5. restringir bypass e push direto conforme a política da equipe.

## Estado do CD

O artefato implantável é a imagem definida em `docker/api/Dockerfile`; o fluxo de
promoção e rollback rastreável está documentado em [`PRODUCTION.md`](PRODUCTION.md).
O repositório ainda não define ambiente de destino, registry de imagens, credenciais,
TLS ou estratégia de rollout. Por segurança, o workflow constrói e testa a imagem, mas
não a publica nem executa um deploy fictício. Quando esses contratos forem decididos,
a entrega deve entrar em job separado, dependente de `quality-gate`, usando GitHub
Environment protegido, credenciais de curta duração e aprovação compatível com a
governança do modelo.

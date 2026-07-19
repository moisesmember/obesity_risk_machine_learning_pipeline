# Ingestão governada do dataset Kaggle

## Objetivo

O comando `obesity-ingest-kaggle` baixa o dataset configurado, valida seu contrato e
publica um snapshot raw imutável. O código de domínio não depende diretamente do
Kaggle: o acesso externo fica no adaptador `KaggleApiDownloader` e pode ser substituído
por um fake nos testes ou por outro provedor no futuro.

## Fluxo

```text
Kaggle
  -> diretório único de staging
  -> localização de obesity_level.csv
  -> validação de schema, domínios, id, duplicatas e alvo
  -> validação do SHA-256 governado
  -> cópia e manifesto em diretório temporário de publicação
  -> rename atômico para data/raw/obesity_risk_dataset/<sha256>/
  -> remoção do staging
```

Downloads parciais e arquivos inválidos nunca são publicados em `data/raw`. Uma nova
execução com o mesmo hash verifica e reutiliza o snapshot existente sem chamar a API.
Nenhum snapshot anterior é sobrescrito.

## Autenticação

Configure a autenticação conforme o cliente oficial do Kaggle. As opções suportadas
pelo ambiente incluem:

- variável `KAGGLE_API_TOKEN`;
- variáveis legadas `KAGGLE_USERNAME` e `KAGGLE_KEY`;
- arquivo oficial `kaggle.json` no diretório de configuração do usuário.

Use `.env.example` apenas como referência. Credenciais reais pertencem ao `.env` local,
ao ambiente do processo ou a um secret manager e nunca devem ser versionadas.

## Execução

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
obesity-ingest-kaggle
```

O snapshot esperado no contrato atual é:

```text
data/raw/obesity_risk_dataset/
└── 04549179841220e7537ee9065fac9cf9446c6368133882b7199a5618ea541ee6/
    ├── obesity_level.csv
    └── manifest.json
```

## Manifesto

O manifesto contém somente metadados não sensíveis:

- provedor, slug e URL da fonte;
- timestamp UTC da ingestão;
- versão identificada pelo SHA-256;
- nome, hash e tamanho do arquivo;
- schema e nome raw do alvo;
- contagem de linhas e distribuição raw do alvo.

Ele não contém credenciais nem exemplos de pessoas.

## Atualização intencional da fonte

Se o Kaggle publicar um arquivo diferente, o comando falhará informando o hash esperado
e o observado. Não desabilite essa validação. Para adotar a nova versão:

1. baixe-a em staging isolado;
2. revise licença, schema, cardinalidades, target e qualidade;
3. atualize o contexto de negócio e o contrato executável;
4. registre o novo SHA-256 em configuração;
5. adicione ou ajuste testes de regressão;
6. execute novamente a ingestão, que criará outro diretório imutável.

Também é possível informar um hash já revisado explicitamente:

```bash
obesity-ingest-kaggle --expected-sha256 <SHA256_REVISADO>
```

## Falhas esperadas

O comando encerra sem publicar quando houver falha de autenticação ou download,
arquivo ausente ou repetido, alteração de schema, campo vazio, categoria desconhecida,
valor fora de domínio, `id` duplicado, classe do alvo ausente ou divergência de hash.

# Obesity Risk Machine Learning Pipeline

Pipeline de Machine Learning para estudar e classificar níveis de risco de obesidade
a partir de características físicas, hábitos alimentares e aspectos do estilo de vida.
O projeto usa como fonte de dados o
[Obesity Risk Dataset, publicado no Kaggle](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset).

> Este projeto tem finalidade educacional e analítica. As previsões geradas por seus
> modelos não constituem diagnóstico médico nem substituem a avaliação de um
> profissional de saúde.

## Documentação do projeto

- [Contexto de negócio e contrato do dataset](docs/BUSINESS_CONTEXT.md): descreve o
  problema, a proveniência, todas as colunas, cardinalidades, distribuições, padrões,
  alvo, riscos e decisões pendentes da pipeline.
- [Ingestão governada do Kaggle](docs/INGESTION.md): explica autenticação, validação,
  publicação imutável, manifesto, idempotência e atualização da fonte.

## Contexto do problema

A obesidade é influenciada por diferentes fatores físicos e comportamentais. Nesse
contexto, uma pipeline de classificação pode ajudar a investigar como esses fatores se
relacionam com diferentes níveis de obesidade, comparar algoritmos e produzir
experimentos reproduzíveis.

O problema será tratado como uma **classificação supervisionada multiclasse**. A
variável-alvo representa o nível de obesidade do indivíduo, enquanto as variáveis de
entrada abrangem quatro grupos principais:

| Grupo | Exemplos de atributos |
| --- | --- |
| Dados demográficos e físicos | gênero, idade, altura e peso |
| Histórico familiar | histórico familiar de sobrepeso |
| Hábitos alimentares | consumo de alimentos calóricos e vegetais, quantidade de refeições, consumo entre refeições, água e álcool |
| Estilo de vida | tabagismo, monitoramento de calorias, atividade física, tempo de uso de dispositivos e meio de transporte |

Uma publicação que utiliza essa mesma fonte descreve 20.758 registros, 16 variáveis de
entrada e sete classes de saída: peso insuficiente, peso normal, sobrepeso níveis I e
II e obesidade tipos I, II e III. Consulte a
[descrição do estudo](https://www.nature.com/articles/s41598-025-20505-9) e confirme
essas características contra a versão efetivamente obtida no Kaggle, pois a fonte pode
ser atualizada.

O trabalho esperado inclui exploração e validação dos dados, preparação das features,
treinamento, otimização, avaliação e rastreamento dos experimentos. O schema, os nomes
das colunas, os tipos e os valores da variável-alvo devem sempre ser validados a partir
do arquivo efetivamente baixado, antes do treinamento.

### Escopo e limitações

- O dataset é tabular e representa associações entre atributos observados e o nível de
  obesidade; ele não deve ser interpretado como evidência causal ou previsão temporal
  de que uma pessoa desenvolverá obesidade.
- Altura e peso podem ter relação direta com a definição da classe por índice de massa
  corporal. O projeto deve auditar essa relação e comparar cenários com e sem essas
  variáveis quando o objetivo for estimar suscetibilidade, evitando uma avaliação
  artificialmente otimista.
- Resultados devem ser analisados por classe e, quando possível, por subgrupos. Um bom
  resultado no dataset não demonstra generalização clínica ou populacional.
- Versão, hash, schema, quantidade de registros e regras de exclusão do arquivo baixado
  devem ser registrados para tornar cada experimento reproduzível.

## Objetivos

- Construir uma pipeline reproduzível para preparação dos dados, treinamento e
  inferência.
- Comparar modelos de classificação sob o mesmo contrato de dados e validação.
- Evitar vazamento de dados mantendo o ajuste das transformações restrito ao conjunto
  de treino.
- Avaliar o desempenho por classe com métricas adequadas a um problema multiclasse.
- Rastrear parâmetros, métricas e artefatos dos experimentos com MLflow.
- Armazenar metadados no PostgreSQL e artefatos no MinIO durante o desenvolvimento
  local.

## Tecnologias

- Python, Pandas, NumPy e Scikit-learn para processamento e modelagem.
- JupyterLab para exploração e execução dos notebooks.
- Optuna para otimização de hiperparâmetros.
- MLflow para rastreamento de experimentos.
- PostgreSQL como backend de metadados do MLflow.
- MinIO como armazenamento de artefatos compatível com S3.
- Docker Compose para orquestrar a infraestrutura local.
- AutoGluon, H2O e FLAML como benchmarks opcionais.

## Pré-requisitos

Antes de iniciar, instale:

- Python compatível com as dependências declaradas no projeto;
- Git;
- Docker Desktop, no Windows ou macOS, ou Docker Engine com o plugin Compose, no
  Linux;
- uma conta no Kaggle para baixar o dataset pela interface web ou pela CLI.

Confirme as instalações:

```bash
python --version
docker version
docker compose version
```

No Windows, se o comando `python` não estiver disponível, use `py` nos comandos de
criação do ambiente virtual.

## Instalação

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd obesity_risk_machine_learning_pipeline
```

### 2. Crie o ambiente virtual

No Windows, usando PowerShell:

```powershell
py -m venv .venv
```

No Linux ou macOS:

```bash
python3 -m venv .venv
```

### 3. Ative o ambiente virtual

No Windows, usando PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear scripts locais, libere-os apenas para a sessão atual e tente
novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

No Prompt de Comando do Windows:

```bat
.venv\Scripts\activate.bat
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Quando o ambiente estiver ativo, o terminal normalmente exibirá `(.venv)` antes do
prompt. Para sair dele, execute `deactivate`.

### 4. Instale as dependências

Atualize o instalador e instale as dependências principais:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Para executar também os benchmarks opcionais:

```bash
python -m pip install -r requirements-benchmarks.txt
```

O arquivo de benchmarks já inclui as dependências principais. AutoGluon e H2O possuem
melhor suporte em Linux ou WSL; por isso, a instalação opcional pode exigir ajustes
adicionais no Windows.

## Obtenção dos dados

Configure a credencial do Kaggle em `KAGGLE_API_TOKEN`, nas variáveis legadas
`KAGGLE_USERNAME` e `KAGGLE_KEY` ou no arquivo oficial `kaggle.json`. Nunca versione a
credencial; use [.env.example](.env.example) apenas como referência.

Com o ambiente virtual ativo e o pacote instalado, inicialize os dados:

```bash
obesity-initialize
```

Se o snapshot já existir, o comando valida sua integridade e pula o download. Se não
existir, baixa em staging isolado, valida schema, domínios, identificadores, duplicatas,
classes e SHA-256 e somente então publica:

```text
data/raw/obesity_risk_dataset/<sha256>/
├── obesity_level.csv
└── manifest.json
```

O raw é imutável e uma repetição reutiliza o mesmo snapshot sem sobrescrevê-lo. Se a
fonte mudar, a ingestão falhará por divergência de hash até que a nova versão seja
auditada e o contrato seja atualizado intencionalmente. Consulte o
[guia de ingestão](docs/INGESTION.md) para detalhes.

## Infraestrutura local com Docker Compose

O arquivo `docker-compose.yml` define três serviços:

| Serviço | Finalidade | Acesso local |
| --- | --- | --- |
| PostgreSQL | Backend de metadados do MLflow | `localhost:5432` |
| MinIO | Armazenamento dos artefatos | API em `http://localhost:9000` e console em `http://localhost:9001` |
| MLflow | Rastreamento dos experimentos | `http://localhost:5000` |

Com o Docker em execução, suba os serviços em segundo plano:

```bash
docker compose up --build -d
```

Confira o estado e acompanhe os logs:

```bash
docker compose ps
docker compose logs -f
```

Para interromper os serviços sem apagar os volumes persistentes:

```bash
docker compose down
```

As credenciais padrão presentes no Compose servem somente para desenvolvimento local.
Em ambientes compartilhados, configure valores seguros por variáveis de ambiente e
nunca versione segredos.

> **Estado atual:** o Compose referencia `docker/mlflow/Dockerfile`, mas esse arquivo
> ainda não existe no repositório. O build do serviço MLflow somente funcionará depois
> que essa imagem for adicionada. O destino S3 configurado também pressupõe a existência
> prévia do bucket usado pelo MLflow no MinIO.

## Execução dos notebooks

Com o ambiente virtual ativo e as dependências instaladas, inicie o JupyterLab a partir
da raiz do projeto:

```bash
python -m jupyter lab
```

O navegador abrirá a interface local do Jupyter. Acesse o diretório `notebooks/`, abra
o notebook desejado e selecione o kernel Python associado ao ambiente `.venv`.

Para executar um notebook completo pela linha de comando, depois que ele existir no
projeto, use:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/<NOME_DO_NOTEBOOK>.ipynb --output <NOME_DO_NOTEBOOK>_executado.ipynb
```

> O repositório ainda não contém notebooks versionados. Crie o diretório `notebooks/`
> para os trabalhos exploratórios e mantenha a lógica reutilizável da pipeline em
> módulos Python, em vez de deixá-la exclusivamente no notebook.

## Fluxo rápido de inicialização

Depois da primeira instalação, o fluxo usual de desenvolvimento é:

```powershell
.\.venv\Scripts\Activate.ps1
obesity-initialize
docker compose up --build -d
python -m jupyter lab
```

Ao finalizar:

```powershell
docker compose down
deactivate
```

## Boas práticas para o projeto

- Fixe seeds e registre versões de dados, código, parâmetros e bibliotecas.
- Separe treino, validação e teste antes de ajustar transformações ou realizar
  amostragem.
- Não use o conjunto de teste para escolher modelo, hiperparâmetros ou threshold.
- Salve preprocessamento e modelo em uma única pipeline para manter paridade entre
  treino e inferência.
- Analise métricas por classe e matriz de confusão; não dependa apenas de accuracy.
- Não versione credenciais, dados sensíveis, artefatos grandes ou arquivos brutos sem
  uma política explícita.

## Licença

O código deste repositório é distribuído conforme o arquivo [LICENSE](LICENSE). O
dataset possui termos próprios, publicados separadamente em sua página no Kaggle.

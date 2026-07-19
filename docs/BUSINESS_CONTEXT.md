# Contexto de negócio e contrato do dataset

## 1. Finalidade deste documento

Este documento define o entendimento de negócio, o contrato inicial dos dados e os
principais riscos de modelagem do projeto **Obesity Risk Machine Learning Pipeline**.
Ele deve orientar a exploração, a validação, o treinamento, a avaliação e a inferência,
sem substituir contratos executáveis no código.

As estatísticas apresentadas foram calculadas diretamente sobre o arquivo
`obesity_level.csv`, obtido do
[Obesity Risk Dataset no Kaggle](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)
em 19 de julho de 2026.

| Item | Valor observado |
| --- | --- |
| Arquivo | `obesity_level.csv` |
| Tamanho | 2.444.336 bytes |
| SHA-256 | `04549179841220E7537EE9065FAC9CF9446C6368133882B7199A5618EA541EE6` |
| Registros | 20.758 |
| Colunas | 18: um identificador, 16 atributos preditores e um alvo |
| Valores ausentes | 0 |
| Linhas duplicadas | 0, inclusive quando `id` é desconsiderado |

Os números são válidos para esse snapshot. Uma nova versão do dataset deve receber
novo hash, novo perfil e nova validação antes de ser comparada com experimentos
anteriores.

## 2. Problema de negócio

O dataset reúne características demográficas, antropométricas, alimentares e
comportamentais de indivíduos. O problema analítico é estimar, para um registro, uma
entre sete categorias de estado de peso: peso insuficiente, peso normal, dois níveis de
sobrepeso e três tipos de obesidade.

Apesar de a fonte usar o termo **risco de obesidade**, o alvo representa o nível de
obesidade associado ao estado observado, e não a ocorrência futura de um evento. Não
há data de referência, histórico longitudinal ou janela de acompanhamento. Portanto, a
tarefa inicial deve ser descrita tecnicamente como **classificação multiclasse do estado
de peso**, e não como previsão temporal de desenvolvimento de obesidade.

O modelo pode apoiar estudos metodológicos, comparação de algoritmos, análise de
padrões e protótipos de triagem. Ele não deve ser apresentado como diagnóstico médico,
prescrição, recomendação de tratamento ou evidência de causalidade.

### 2.1 Unidade de análise

Cada linha representa um perfil individual com:

- características pessoais: gênero e idade;
- medidas corporais: altura e peso;
- histórico familiar de sobrepeso;
- hábitos de alimentação e hidratação;
- tabagismo, atividade física e monitoramento de calorias;
- uso de dispositivos e meio de transporte;
- uma classe associada ao nível de obesidade.

O arquivo não contém identificador de pessoa reutilizável, data, local, instituição ou
grupo familiar. O campo `id` identifica somente a linha e não prova independência entre
indivíduos.

### 2.2 Saída esperada da pipeline

Para cada entrada válida, a inferência deve produzir:

- a classe prevista em sua nomenclatura canônica;
- a probabilidade ou score de cada uma das sete classes;
- a confiança da decisão, quando definida uma política de confiança;
- a versão do modelo, do schema e do feature set usados;
- um identificador rastreável da execução, sem expor dados pessoais.

Uma regra de encaminhamento, um agrupamento entre classes ou um threshold operacional
só poderá ser definido depois que o responsável de negócio informar a ação associada à
previsão e os custos de cada tipo de erro.

## 3. Proveniência e representatividade

A estrutura das variáveis é derivada do dataset
[Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition),
publicado no UCI Machine Learning Repository. A fonte original contém 2.111 registros
de pessoas de México, Peru e Colômbia; informa ainda que 23% foram coletados por uma
plataforma web e 77% foram gerados sinteticamente com Weka e SMOTE.

O arquivo usado neste projeto contém 20.758 registros, número muito superior ao da
fonte original. A página do Kaggle não documenta de forma suficiente toda a cadeia de
transformações que gerou essa versão. Consequentemente:

- a origem geográfica da base UCI não deve ser automaticamente atribuída a todos os
  registros deste snapshot;
- os registros não devem ser descritos como pacientes clínicos sem comprovação;
- padrões do dataset podem refletir geração ou balanceamento sintético, e não a
  prevalência real de uma população;
- resultados precisam de validação externa antes de qualquer aplicação no mundo real.

Essa limitação de linhagem é um risco de governança. A pipeline deve registrar o slug
do Kaggle, o nome do arquivo, o hash, a data de obtenção e as estatísticas de schema em
cada run.

## 4. Visão geral do contrato de dados

| Grupo | Colunas raw | Papel |
| --- | --- | --- |
| Identificação | `id` | Rastrear a linha na origem; nunca usar como feature |
| Demografia | `Gender`, `Age` | Caracterização individual e auditoria de subgrupos |
| Antropometria | `Height`, `Weight` | Medidas corporais fortemente relacionadas ao alvo |
| Histórico familiar | `family_history_with_overweight` | Indício declarado de predisposição familiar |
| Alimentação | `FAVC`, `FCVC`, `NCP`, `CAEC`, `CH2O`, `CALC` | Hábitos alimentares e de hidratação |
| Condição e estilo de vida | `SMOKE`, `SCC`, `FAF`, `TUE`, `MTRANS` | Hábitos e condições comportamentais |
| Alvo | `0be1dad` | Classe de estado de peso; nome raw contém erro de grafia |

## 5. Dicionário de atributos

### 5.1 Identificação, demografia e antropometria

| Coluna raw | Tipo observado | Cardinalidade | Domínio observado | Significado e regra de negócio |
| --- | --- | ---: | --- | --- |
| `id` | inteiro | 20.758 | 0 a 20.757, sequencial e único | Identificador técnico da linha. Deve ser preservado na rastreabilidade e excluído antes do fit, pois não contém sinal de negócio legítimo. |
| `Gender` | categórico | 2 | `Female`, `Male` | Gênero registrado na fonte. Deve ser tratado como categoria nominal e usado em auditorias de desempenho e equidade. A fonte não oferece outras categorias nem esclarece se representa gênero autodeclarado ou sexo biológico. |
| `Age` | numérico contínuo | 1.703 | 14 a 61 anos | Idade. Embora o conceito seja contado em anos, 49,35% dos valores possuem casas decimais, o que sugere interpolação ou geração sintética. Não arredondar silenciosamente. |
| `Height` | numérico contínuo | 1.833 | 1,45 a 1,9757 m | Altura em metros. Deve ser positiva, plausível e informada no momento da decisão. É usada no cálculo de IMC e tem relação estrutural com o alvo. |
| `Weight` | numérico contínuo | 1.979 | 39 a 165,0573 kg | Peso em quilogramas. É uma feature altamente preditiva e potencial proxy direto da classe. Deve ser validada quanto à unidade e disponibilidade na inferência. |

### 5.2 Histórico familiar e alimentação

| Coluna raw | Tipo observado | Cardinalidade | Domínio observado | Significado e regra de negócio |
| --- | --- | ---: | --- | --- |
| `family_history_with_overweight` | binário inteiro | 2 | `0`, `1` | Indica se um familiar sofreu ou sofre de sobrepeso. Normalizar para booleano: `0 = não`, `1 = sim`. |
| `FAVC` | binário inteiro | 2 | `0`, `1` | *Frequent consumption of high caloric food*: consumo frequente de alimentos com alto valor calórico. Normalizar para booleano. |
| `FCVC` | numérico ordinal/contínuo | 934 | 1 a 3 | *Frequency of consumption of vegetables*: frequência de consumo de vegetais. A escala original é ordinal, mas o snapshot contém valores fracionários; preservar como numérico durante a análise e documentar qualquer discretização. |
| `NCP` | numérico ordinal/contínuo | 689 | 1 a 4 | *Number of main meals*: número ou nível associado à quantidade de refeições principais diárias. Valores fracionários aparecem no snapshot e não devem ser interpretados literalmente sem regra validada. |
| `CAEC` | categórico ordinal | 4 | `0`, `Sometimes`, `Frequently`, `Always` | *Consumption of food between meals*: frequência de consumo entre refeições. O valor raw `0` representa a categoria negativa da fonte e deve ser normalizado para `No`. |
| `CH2O` | numérico ordinal/contínuo | 1.506 | 1 a 3 | Consumo diário de água em uma escala de três níveis. O nome contém a letra `O`, não o algarismo zero. O snapshot possui valores fracionários. |
| `CALC` | categórico ordinal | 3 | `0`, `Sometimes`, `Frequently` | Frequência de consumo de álcool. O valor raw `0` deve ser normalizado para `No`. A categoria `Always`, existente no instrumento original, não aparece neste snapshot e deve ser tratada como categoria conhecida porém não observada. |

### 5.3 Condição física e estilo de vida

| Coluna raw | Tipo observado | Cardinalidade | Domínio observado | Significado e regra de negócio |
| --- | --- | ---: | --- | --- |
| `SMOKE` | binário inteiro | 2 | `0`, `1` | Indica hábito de fumar: `0 = não`, `1 = sim`. A classe positiva é muito rara. |
| `SCC` | binário inteiro | 2 | `0`, `1` | *Calories consumption monitoring*: indica monitoramento do consumo de calorias. Não significa consumo de bebida calórica. |
| `FAF` | numérico ordinal/contínuo | 1.360 | 0 a 3 | *Physical activity frequency*: frequência de atividade física em uma escala de quatro níveis. Valores fracionários indicam interpolação ou geração. |
| `TUE` | numérico ordinal/contínuo | 1.297 | 0 a 2 | *Time using technology devices*: nível de tempo dedicado a dispositivos tecnológicos. Não interpretar diretamente como horas sem recuperar o instrumento de coleta. |
| `MTRANS` | categórico nominal | 5 | `Public_Transportation`, `Automobile`, `Walking`, `Motorbike`, `Bike` | Principal meio de transporte. Categorias raras precisam de tratamento consistente entre treino e inferência. |

### 5.4 Alvo

| Coluna raw | Nome canônico | Tipo | Cardinalidade | Regra |
| --- | --- | --- | ---: | --- |
| `0be1dad` | `NObeyesdad` | categórico | 7 | Renomear na camada de padronização. O primeiro caractere raw é o algarismo zero. |

O alvo também contém o valor incorreto `0rmal_Weight`, com zero no lugar das letras
iniciais. Ele deve ser normalizado para `Normal_Weight`. A correção deve ser explícita,
testada e registrada; o arquivo raw não deve ser alterado.

Ordem semântica de referência:

1. `Insufficient_Weight` — peso insuficiente;
2. `Normal_Weight` — peso normal;
3. `Overweight_Level_I` — sobrepeso nível I;
4. `Overweight_Level_II` — sobrepeso nível II;
5. `Obesity_Type_I` — obesidade tipo I;
6. `Obesity_Type_II` — obesidade tipo II;
7. `Obesity_Type_III` — obesidade tipo III.

Embora exista uma ordem de severidade, a primeira implementação deve manter o contrato
como classificação multiclasse. Uma codificação ordinal pode ser avaliada como
challenger, desde que erros entre classes distantes recebam análise específica e que a
classe de peso insuficiente não seja interpretada como ausência de risco de saúde.

## 6. Perfil estatístico do snapshot

### 6.1 Variáveis numéricas

| Atributo | Únicos | Média | Mínimo | P25 | Mediana | P75 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Age` | 1.703 | 23,8418 | 14,0000 | 20,0000 | 22,8154 | 26,0000 | 61,0000 |
| `Height` | 1.833 | 1,7002 | 1,4500 | 1,6319 | 1,7000 | 1,7629 | 1,9757 |
| `Weight` | 1.979 | 87,8878 | 39,0000 | 66,0000 | 84,0649 | 111,6006 | 165,0573 |
| `FCVC` | 934 | 2,4459 | 1,0000 | 2,0000 | 2,3938 | 3,0000 | 3,0000 |
| `NCP` | 689 | 2,7613 | 1,0000 | 3,0000 | 3,0000 | 3,0000 | 4,0000 |
| `CH2O` | 1.506 | 2,0294 | 1,0000 | 1,7920 | 2,0000 | 2,5496 | 3,0000 |
| `FAF` | 1.360 | 0,9817 | 0,0000 | 0,0080 | 1,0000 | 1,5874 | 3,0000 |
| `TUE` | 1.297 | 0,6168 | 0,0000 | 0,0000 | 0,5739 | 1,0000 | 2,0000 |

A população observada é concentrada em adultos jovens: a mediana de idade é 22,82
anos e 99% dos registros têm até 41 anos. Métricas obtidas nessa base não sustentam,
sem validação externa, generalização para idosos ou outras populações.

### 6.2 Variáveis categóricas e binárias

| Atributo | Distribuição observada |
| --- | --- |
| `Gender` | `Female`: 10.422 (50,21%); `Male`: 10.336 (49,79%) |
| `family_history_with_overweight` | `1`: 17.014 (81,96%); `0`: 3.744 (18,04%) |
| `FAVC` | `1`: 18.982 (91,44%); `0`: 1.776 (8,56%) |
| `CAEC` | `Sometimes`: 17.529 (84,44%); `Frequently`: 2.472 (11,91%); `Always`: 478 (2,30%); `0`: 279 (1,34%) |
| `SMOKE` | `0`: 20.513 (98,82%); `1`: 245 (1,18%) |
| `SCC` | `0`: 20.071 (96,69%); `1`: 687 (3,31%) |
| `CALC` | `Sometimes`: 15.066 (72,58%); `0`: 5.163 (24,87%); `Frequently`: 529 (2,55%) |
| `MTRANS` | `Public_Transportation`: 16.687 (80,39%); `Automobile`: 3.534 (17,02%); `Walking`: 467 (2,25%); `Motorbike`: 38 (0,18%); `Bike`: 32 (0,15%) |

Há categorias extremamente raras. Um split simples pode deixar poucos casos de
`Motorbike`, `Bike`, fumantes ou pessoas que monitoram calorias em alguma partição. A
pipeline deve verificar cobertura por partição e usar encoding tolerante apenas a
categorias declaradas. Categorias desconhecidas em produção devem gerar telemetria, não
ser silenciosamente tratadas como a categoria mais comum.

### 6.3 Distribuição do alvo

| Classe canônica | Valor raw | Registros | Percentual | IMC médio derivado |
| --- | --- | ---: | ---: | ---: |
| `Insufficient_Weight` | `Insufficient_Weight` | 2.523 | 12,15% | 17,58 |
| `Normal_Weight` | `0rmal_Weight` | 3.082 | 14,85% | 22,00 |
| `Overweight_Level_I` | `Overweight_Level_I` | 2.427 | 11,69% | 26,06 |
| `Overweight_Level_II` | `Overweight_Level_II` | 2.522 | 12,15% | 28,19 |
| `Obesity_Type_I` | `Obesity_Type_I` | 2.910 | 14,02% | 32,15 |
| `Obesity_Type_II` | `Obesity_Type_II` | 3.248 | 15,65% | 36,52 |
| `Obesity_Type_III` | `Obesity_Type_III` | 4.046 | 19,49% | 41,78 |

A maior classe possui 1,67 vez o volume da menor. O desbalanceamento não é extremo,
mas torna accuracy isolada inadequada para decidir promoção. O IMC da tabela foi
calculado apenas para auditoria como `Weight / Height²`; ele não existe como coluna no
arquivo.

## 7. Padrões relevantes para o negócio

### 7.1 Altura, peso, IMC e circularidade do alvo

O peso tem correlação de 0,941 com o IMC derivado, e o IMC médio cresce de forma quase
monotônica entre as classes. Isso é esperado porque a rotulagem histórica das
categorias de obesidade utiliza medidas relacionadas ao IMC.

Esse comportamento cria dois cenários de negócio diferentes:

1. **Reproduzir a classificação atual de estado de peso:** altura e peso são válidos,
   mas um cálculo determinístico de IMC é um baseline obrigatório. Um modelo complexo
   só se justifica se melhorar um objetivo claramente definido.
2. **Estimar suscetibilidade por hábitos antes de medir o estado corporal:** altura e,
   principalmente, peso podem tornar a solução circular. Nesse cenário, deve existir
   um challenger sem `Weight`, sem IMC derivado e possivelmente sem `Height`.

Uma métrica muito alta com altura e peso não deve ser celebrada antes dessa auditoria.
Ela pode apenas demonstrar que o modelo reaprendeu a regra usada para formar o rótulo.

### 7.2 Associações fortes com gênero

O dataset é equilibrado por gênero no total, mas não dentro de todas as classes:

| Classe | Feminino | Masculino |
| --- | ---: | ---: |
| `Insufficient_Weight` | 64,25% | 35,75% |
| `Normal_Weight` | 53,86% | 46,14% |
| `Overweight_Level_I` | 44,09% | 55,91% |
| `Overweight_Level_II` | 29,94% | 70,06% |
| `Obesity_Type_I` | 43,54% | 56,46% |
| `Obesity_Type_II` | 0,25% | 99,75% |
| `Obesity_Type_III` | 99,88% | 0,12% |

As duas últimas relações são quase determinísticas. Elas podem ser um artefato da
geração sintética e fazer o modelo usar gênero como atalho. São obrigatórias:

- métricas por gênero e por classe;
- comparação de modelo com e sem `Gender`;
- análise de importância e explicabilidade;
- avaliação externa antes de qualquer uso com pessoas reais.

### 7.3 Outros padrões e desequilíbrios

- Histórico familiar positivo aparece em 81,96% da base, em 99,85% da classe
  `Obesity_Type_II` e em 99,98% da `Obesity_Type_III`.
- Consumo frequente de alimentos calóricos é positivo em 91,44% da base, limitando o
  poder de comparação da categoria negativa.
- Apenas 1,18% dos registros indicam tabagismo e 3,31% indicam monitoramento de
  calorias; métricas específicas para esses grupos terão alta incerteza.
- Transporte público representa 80,39% dos registros. `Bike` e `Motorbike` somam apenas
  70 casos e exigem cuidado em validação cruzada.
- Várias escalas originalmente ordinais aparecem como números contínuos. Isso sugere
  dados interpolados ou sintéticos e impede interpretar cada decimal como uma resposta
  literal de questionário.

Esses padrões são associações internas do snapshot. Eles não demonstram causa, não
representam prevalência populacional e não devem gerar recomendações individuais de
saúde.

## 8. Regras de qualidade e normalização

### 8.1 Normalizações obrigatórias

| Campo | Valor raw | Valor canônico |
| --- | --- | --- |
| Nome do alvo | `0be1dad` | `NObeyesdad` |
| Classe normal | `0rmal_Weight` | `Normal_Weight` |
| `CAEC` negativo | `0` | `No` |
| `CALC` negativo | `0` | `No` |
| Binários | `0`, `1` | booleano ou enum `No`, `Yes`, conforme contrato interno |

As transformações pertencem à camada de staging/validação. O arquivo raw deve
permanecer imutável.

### 8.2 Validações bloqueantes

A ingestão deve falhar com mensagem acionável quando ocorrer:

- ausência de qualquer coluna obrigatória;
- coluna adicional não autorizada pelo versionamento de schema;
- `id` ausente, repetido ou fora da política definida;
- alvo ausente ou fora das sete classes conhecidas;
- valores binários diferentes de `0` e `1`;
- categoria desconhecida sem política explícita;
- idade, altura ou peso ausente, não numérico, não finito ou fora do domínio aceito;
- valores fora das escalas `FCVC`, `NCP`, `CH2O`, `FAF` e `TUE`;
- duplicata de registro desconsiderando `id`;
- alteração inesperada de cardinalidade ou perda de uma classe do alvo.

Os limites observados neste snapshot não são automaticamente limites clínicos. Eles
servem como contrato inicial e devem ser separados entre erro impossível, valor
plausível novo e drift que exige revisão.

## 9. Desenho recomendado da pipeline

```text
Kaggle
  -> download em staging isolado
  -> hash e manifesto da fonte
  -> validação do schema raw
  -> normalização de nomes e categorias
  -> auditorias de qualidade e leakage
  -> split estratificado
  -> fit de preprocessamento somente no treino
  -> baseline determinístico de IMC e baseline estatístico
  -> seleção e otimização somente em treino/validação
  -> avaliação única no teste intocado
  -> auditorias por classe e subgrupo
  -> model card, manifesto e MLflow
  -> artefato completo para inferência
```

### 9.1 Split e validação

Não existem coluna temporal nem identificador de grupo. Para o objetivo experimental
atual, um split aleatório estratificado é a opção disponível, com seed fixa e
preservação das sete classes. Antes de aceitá-lo, a pipeline deve:

- procurar registros idênticos e quase duplicados;
- garantir cobertura das categorias raras;
- separar o teste antes de qualquer amostragem, imputação ou transformação com fit;
- impedir que teste e validação orientem hiperparâmetros ou escolha do candidato.

Esse split estima generalização apenas para registros intercambiáveis com o mesmo
processo de geração. Ele não mede generalização temporal, geográfica ou clínica.

### 9.2 Preprocessamento

- Remover `id` das features e preservá-lo somente para rastreabilidade.
- Corrigir os valores raw conhecidos antes da validação canônica.
- Tratar `Gender`, `CAEC`, `CALC` e `MTRANS` como categóricos.
- Tratar binários de forma consistente e com schema explícito.
- Manter as escalas fracionárias como numéricas no baseline; discretizações devem ser
  challengers comparados na validação.
- Ajustar encoders, escaladores, seleção e imputação somente no treino.
- Serializar preprocessamento e estimador em uma única pipeline.

### 9.3 Baselines obrigatórios

1. Regra simples baseada no IMC derivado, para medir quanto do alvo é explicado pela
   definição antropométrica.
2. Classificador ingênuo pela classe majoritária ou distribuição estratificada.
3. Modelo linear multiclasse com preprocessamento completo.
4. Modelo simples baseado em árvores.
5. Challenger sem peso e sem IMC, orientado aos hábitos e histórico familiar.

Modelos complexos e AutoML devem superar baselines sob o mesmo split e o mesmo contrato
sem consultar o teste durante a seleção.

### 9.4 Métricas

Enquanto custos de negócio não forem definidos, recomenda-se provisoriamente:

- **métrica principal:** macro F1, por dar peso equivalente às sete classes;
- balanced accuracy e recall por classe;
- matriz de confusão, com destaque para erros entre classes distantes;
- log loss e calibração das probabilidades;
- métricas por gênero e demais segmentos com amostra suficiente;
- latência, tamanho do artefato e taxa de entradas rejeitadas na inferência.

Accuracy deve ser apenas secundária. Não há base de negócio para fixar threshold,
recall mínimo ou gate de promoção numérico neste momento.

## 10. Riscos e controles

| Risco | Impacto | Controle esperado |
| --- | --- | --- |
| Circularidade por altura e peso | Métrica artificialmente alta e baixa utilidade para prevenção | Baseline de IMC e comparação com modelo sem peso/IMC |
| Linhagem incompleta da versão Kaggle | Resultados não reproduzíveis ou representatividade mal descrita | Hash, manifesto, snapshot e documentação de proveniência |
| Dados sintéticos ou interpolados | Padrões artificiais e generalização limitada | Validação externa e proibição de alegações clínicas |
| Atalho por gênero | Discriminação e degradação fora da amostra | Métricas por grupo, ablação e revisão humana |
| Categorias raras | Estimativas instáveis e falha de encoding | Cobertura por partição e política para desconhecidos |
| Alvo e categoria com typo | Quebra de contrato e duplicação semântica | Mapeamento canônico versionado e testado |
| Ausência de tempo e grupo | Split incapaz de medir generalização futura | Limitar claims; obter nova fonte para validação temporal/externa |
| Uso como diagnóstico | Dano ao usuário e interpretação indevida | Avisos, escopo educacional e aprovação humana |

## 11. Monitoramento de inferência

Se o modelo for exposto por API ou batch, monitorar:

- conformidade do schema e taxa de rejeição;
- valores ausentes, inválidos e categorias desconhecidas;
- drift de distribuição por atributo e subgrupo;
- distribuição das classes e das probabilidades previstas;
- baixa confiança, entropia das probabilidades e mudanças no top-2;
- desempenho por classe e gênero quando labels posteriores estiverem disponíveis;
- versão do dataset, pipeline, modelo e threshold utilizados.

Sem labels reais coletados após a implantação, só será possível monitorar qualidade de
entrada e drift, não desempenho real.

## 12. Decisões de negócio pendentes

As seguintes definições não estão disponíveis na fonte nem no repositório e não devem
ser inventadas:

1. Quem consumirá a previsão e qual decisão concreta ela apoiará?
2. O objetivo é classificar estado atual ou estimar risco futuro?
3. Altura e peso estarão disponíveis no instante da decisão?
4. Qual população, região e faixa etária são alvo da solução?
5. Qual é o custo de confundir cada par de classes?
6. Haverá agrupamento operacional, como risco baixo, moderado e alto?
7. Qual capacidade de atendimento, SLA e latência são necessários?
8. Quais métricas e limites determinam rejeição ou candidatura à promoção?
9. É obrigatória aprovação humana antes de qualquer ação?
10. Qual dataset externo permitirá avaliar generalização e equidade?

Até essas respostas existirem, o projeto deve permanecer como pipeline experimental
governada, sem promoção automática e sem alegação de prontidão clínica.

## 13. Critério de pronto para a primeira versão

A primeira versão da solução estará tecnicamente pronta quando:

- o snapshot e seu hash forem registrados sem alterar os dados raw;
- o schema e as normalizações descritas aqui tiverem testes automatizados;
- o split estratificado ocorrer antes de qualquer fit ou amostragem;
- os baselines com e sem medidas antropométricas forem comparados;
- as métricas globais, por classe e por gênero forem publicadas;
- a pipeline salva reproduzir as previsões após ser recarregada;
- o relatório explicitar limitações, proveniência e riscos de uso;
- os artefatos obrigatórios forem rastreados no MLflow e no manifesto;
- decisões de promoção permanecerem separadas do treinamento.

## 14. Referências

- [Obesity Risk Dataset — Kaggle](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)
- [UCI Machine Learning Repository — Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition)
- [Artigo de apresentação do dataset original](https://doi.org/10.1016/j.dib.2019.104344)
- [Estudo que usa a versão de 20.758 registros](https://www.nature.com/articles/s41598-025-20505-9)

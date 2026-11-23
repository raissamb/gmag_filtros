# Gmag Filtros

Projeto que reúne desenvolvimento de filtros para aplicação em dados potenciais pelos alunos do GMAG UFPA.

## Configurando ambiente de desenvolvimento

O ambiente de desenvolvimento é primariamente em Python, assim é necessário que todos os alunos façam o seguinte:

- Em um pc (Windows ou Linux), instale o Python disponível em [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Escolham um editor/IDE, recomenda-se o VS Code e as seguintes extensões: Pylance, Python, Python Debugger, Python Environments, Rainbow CSV
- Usem a estrutura de diretório presento no arquivo gmag_filtros.zip
- Configure um ambiente virtual para o projeto
- Faça o download dos pacotes necessários informados no arquivo de ambiente requirements.txt

Para configurar um ambiente virtual siga os passos abaixo.

### Configurando Python Env no Windows

Comando para criar o ambiente virtual na pasta raiz do projeto

```shell
python -m venv .venv
```

Para instalar os pacotes necessários para o projeto, primeiro deve-se ativar o ambiente manualmente pelo terminal ou deixando o VS Code ativar. Use o comando abaixo no terminal para ativar o ambiente virtual.

```shell
.\.venv\Scripts\Activate.ps1
```

Você saberá que o ambiente está ativado quando o terminal no VS Code exibir a expressão "(.venv)" como mostrado abaixo

```shell
(.venv) usuario:~/Caminho/gmag_filtros$ 
```

Pode ser que o Windows impeça a ativação do ambiente devido a questões de segurança em relação a execuções de comandos em Shell, se durante a a ativação do ambiente o terminal exiba alguma mensagem de erro onde o ambiente não for ativado, use o comando abaixo para liberar a sessão

```shell
Set-ExecutionPolicy Unrestricted -Scope Process
```

Depois tente ativar o ambiente novamente, deve funcionar agora.

### Configurando Python Env no Linux

Comando para criar o ambiente virtual na pasta raiz do projeto

```shell
python3 -m venv .venv
```

Use o comando abaixo no terminal para ativar o ambiente virtual.

```shell
source .venv/bin/activate
```

### Instalando os pacotes necessários para o projeto

Os pacotes a serem usados nesse projeto são: harmonica, verde, numpy, pandas, matplotlib e seaborn. Pelo menos por enquanto são esses, caso seja necessário instalar outros, faremos depois. Para instalar esses pacotes, existem duas opções.

A opção 1, com o ambiente já ativado, requer o comando

```python
pip install pacote 
```

A opção 2 (a que vamos usar aqui) é instalar os pacotes diretamente de um arquivo de ambiente chamado requirements.txt. Isso é bom porque garante que todos usem exatamente a mesma versão de todos os pacotes. Digite o comando abaixo no terminal

```python
pip install -r requirements.txt
```

Pronto, o ambiente já está configurado, pode começar a trabalhar :)

## Descrição do diretório

Esse projeto contém as seguintes pastas:

- code
- synthetic data
- validation
- data
- output

### Pasta Code

A pasta code é onde fica localizado todos os códigos do trabalho, divididas em subpastas:

- equivalent_layer: onde ficam os códigos principais
  - run_ceq.py: é o script principal que executa todas as tarefas
  - inversion_stage.py: é o script responsável pela construção da camada equivalente
  - transformation_stage: é o script que executa a maioria das transformações
  - poisson_stage: script que contém as instruções para a execução da transformação Razão de Poisson (Poisson ratio)

- frequency: contém o código adaptado do fortran que realiza diversas transformações no domínio da frequencia

- modules: contém os módulos onde as funções que executam diversas tarefas estão
  - constants.py: módulo para guardar valores de constantes, não usado no momento
  - kernel.py: contém as funções do método grav e mag
  - utils.py: contém funções de leitura e escrita de dados e plots

### Pasta synthetic data

Essa pasta contém os scripts que produzem modelos sintéticos.

- synthetic_data_template.py: produz dados sintéticos (grav e mag) em grid regular e superficie de observação plana, N prismas.
- synthetic_data_scattered: produz dados sintéticos (grav e mag) em dados espalhados espacialmente.

### Pasta validation

Aqui está os testes que mostram a validação das funções que realizam as Transformações.

**Sugestão**: para desenvolver sua transformação, faça nesse formato onde um unico arquivo le os dados de entrada e produz a transformação. Assim que o resultado for validado, a sua função será depois incorporado ao código do projeto. Olhe o arquivo gmag_filtros/code/validation/transf_name_validation.py para usar como template.

### Pasta data

Nessa pasta estão os datasets a serem analizados. Cada subpasta aqui deve ser um único dataset. A execução do script run_ceq.py lê essa pasta e os arquivos contidos nela para produzir resultados, assim os resultados estão em gmag_filtros/data/dataset_name.

### Pasta config_templates

Aqui tem templates para criação de arquivos de configuração da camada equivalente e das transformações.

## Equivalent Layer Solvers

The inversion solver is configured via `"solver"` inside `inversion_parameters` in the equivalent layer JSON file.

| Solver | Constraint | Scaling / Speed | When to Use |
| :--- | :--- | :--- | :--- |
| **`cholesky`** *(default)* | Unconstrained ($p \in \mathbb{R}$) | Very fast ($G^T G$ precomputed once) | Gravity data, Poisson ratio, upward/downward continuation, and high-latitude magnetic data. |
| **`bvls`** | Positivity ($p \ge 0$) | Slow for $N > 1000$ (active-set dense solver) | Low-latitude magnetic RTP on small grids ($N \lesssim 1000$) under strictly induced magnetization. |
| **`trf`** | Positivity ($p \ge 0$) | Much faster than `bvls` on medium/large grids | Low-latitude magnetic RTP on larger grids ($N > 1000$) under strictly induced magnetization. |

> **Note on Positivity Constraints (`bvls` and `trf`):** Imposing $p \ge 0$ eliminates neighbor sign oscillations that cause striation artifacts in low-latitude Reduction to the Pole (RTP). It assumes strictly induced magnetization with positive susceptibility contrast and is **not** valid for gravity inversions or bodies with strong remanence.

## Como usar o run_ceq.py

Antes de qualquer coisa, garanta que o terminal esteja aberto (e com ambiente virtual ativado) na pasta onde está o run_ceq.py, ou seja, no terminal deve ter um caminho desse tipo: **seu_caminho/gmag_filtros/code/equivalent_layer**.

Cada dataset deve ser salvo dentro da pasta gmag_filtros/data/**dataset_name**. Nessa pasta deve conter um arquivo com os dados grav e outro com os dados mag. Cada arquivo de dados observados deve seguir a ordem: X,Y,Z,Obs, onde X é Easting, Y é Northing, Z é upwards e Obs é o dado observado. Lembrando que o sistema de coordenadas esperado para a execução da Camada Equivalente Clássica (script run_ceq.py) é o ENU, onde Eixo X é Easting, Y é Northing e Z é Upwards em metros, positivo para cima.

### Exemplo com dataset model2

Vamos usar como exemplo o dataset model2, dentro dessa pasta existe a pasta **input** onde estão os arquivos de dados observados mag e grav, bem como as imagens dos campos e da fonte verdadeira (isso é possível porque o arquivo synthetic_data_template.py ao gerar o dado sintético, já organiza desta forma). Outra pasta que já vai existir aqui caso o dataset tenha sido criado pelo script será a pasta **config**, com um arquivo chamado eq_layer1.json. Nessa pasta ficarão os arquivos de configuração para construção da camada equivalente e das transformações. Você deve preencher o arquivo eq_layer1.json com as informações relevantes para a construção da camada equivalente que você deseja. É possível construir várias camadas, desde que você use diferentes arquivos json para tal, basta copiar e mudar o nome do arquivo e preencher com as informações necessárias.

Representação do diretório de model2:

- data
  - model2
    - config
      - eq_layer1.json
    - input
      - model2_grav.csv
      - model2_mag.csv
      - figuras e metadata file com informações da fonte

Exemplo do arquivo eq_layer1.json para o dataset model2:

```json
{
    "project_settings": {
        "gravity_file": "model2_grav.csv",
        "magnetic_file": "model2_mag.csv"
    },
    "data_columns": {
        "easting_col": "X",
        "northing_col": "Y",
        "upwards_col": "Z",
        "field": "obs"
    },
    "inversion_parameters": {
        "z_top": -200.0,
        "z_bottom": -1000.0,
        "incf": 45.0,
        "decf": 12.0,
        "incs": 45.0,
        "decs": 12.0,
        "reg_min": 1e-06,
        "reg_max": 100.0,
        "steps": 20
    }
}
```

Para executar a construção da Camada Equivalente Clássica (um prisma embaixo de cada ponto de observação), vá na pasta gmag_filtros/code/equivalent_layer/ e encontre o arquivo run_ceq.py, esse é o arquivo que executa esta tarefa. Então, no terminal digite o seguinte comando:

```shell
python run_ceq.py --dataset DATASET_NAME --type DATA_TYPE --eq-config EQ_FILENAME --trans-config TRANSFORMATION_FILENAME --poisson-config POISSON_FILENAME
```

Onde:

- DATASET_NAME é o nome da sua pasta de dados
- DATA_TYPE é o tipo de dado que será processado (grav/gravity ou mag/magnetic)
- EQ_FILENAME é o arquivo de configuração para criar a camada equivalente
- TRANSFORMATION_FILENAME é o arquivo de configuração para uma transformação
- POISSON_FILENAME é o arquivo de configuração específico para a transformação Razão de Poisson

Dê Enter e espera a finalização do processo, uma mensagem aparecerá no terminal quando o processo estiver completo.

**Observação**: você não precisa rodar o comando com todas as flags ativadas de uma única vez, é possível fazer por partes. Por exemplo, caso queira construir só a camada equivalente, use

```shell
python run_ceq.py --dataset DATASET_NAME --type DATA_TYPE --eq-config EQ_FILENAME 
```

Assim, uma pasta com o mesmo nome de EQ_FILENAME será criada para guardar os resultados da camada equivalente do tipo DATA_TYPE que criada com as informações presentes em EQ_FILENAME.

Quando esta tarefa for executada, o diretório da dataset mudará para:

- data
  - model2
    - config
      - eq_layer1.json
    - eq_layer1
      - DATA_TYPE_standard_equivalent_sources_parameters.csv
      - DATA_TYPE_standard_l_curve_metrics
      - figura de comparacao entre campo obs e campo predito pela camada equivalente
    - input
      - model2_grav.csv
      - model2_mag.csv
      - figuras e metadata file com informações da fonte

A pasta **eq_layer1** foi criada para guardar os resultados da camada equivalente criada pelo arquivo de mesmo nome. Assim, toda vez que você criar uma nova camada com nome diferente, uma pasta nova será criada. Isso permite que voce execute transformações para cada camada equivalente de forma isolada. Nessa pasta temos os seguintes arquivos:

- DATA_TYPE_standard_equivalent_sources_parameters.csv: arquivo com as informações dos prismas da camada e o valor da propriedade física estimada
- DATA_TYPE_standard_l_curve_metrics: arquivo com informações da curva L, método usado para resolver o sistema de equações da inversão

Para executar uma transformação, primeiro é preciso criar o arquivo de configuração referente a tal transformação. Na pasta **config_templates** existe alguns exemplos de como fazer tal, abaixo está o arquivo para continuação para cima:

```json
{
    "eq_layer_config": "eq_layer1",
    "transformation_type": "upward_continuation",
    "data_type": "magnetic",
    "execute": true,
    "parameters": {
        "dz": 500
    }
    
}
```

Nesse arquivo tem-se a determinação de:

- eq_layer_config: é o arquivo que contém as informações da camada equivalente que será usada de input para a transformação
- transformation_type: nome da transformação
- data_type: tipo de dado
- execute: flag para confirmar execução da transformação
- parameters: aqui dentro ficam os parametros especificos da transformação, no caso da continuação para cima é a altura do plano de observação desejado

Com uma camada equivalente já construída, para executar a transformação basta digitar no terminal:

```shell
python run_ceq.py --dataset DATASET_NAME --type DATA_TYPE --trans-config TRANSFORMATION_FILENAME 
```

Quando esta tarefa for executada, o diretório da dataset mudará para:

- data
  - model2
    - config
      - eq_layer1.json
    - eq_layer1
      - DATA_TYPE_standard_equivalent_sources_parameters.csv
      - DATA_TYPE_standard_l_curve_metrics
      - figura de comparacao entre campo obs e campo predito pela camada equivalente
      - transformations
        - TRANSFORMATION_FILENAME
          - resultados da transformacao aqui
    - input
      - model2_grav.csv
      - model2_mag.csv
      - figuras e metadata file com informações da fonte

Essa nova organização permite que eu possa fazer diversas transformações para uma única camada.

## A fazer

- Implementar as transformações: Derivadas, HGM, Tilt, ASA e Deconvolução de Euler

- Para o caso da RTP, adicionar o vinculo de positividade para garantir magnetização estimada positiva, isso necessita mudar a forma de resolver o sistema já que Cholesky não será possível nessa situação

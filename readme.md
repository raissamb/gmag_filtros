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

**Sugestão**: para desenvolver sua transformação, faça nesse formato onde um unico arquivo le os dados de entrada e produz a transformação. Assim que o resultado for validado, a sua função será depois incorporado ao código do projeto.

### Pasta data

Nessa pasta estão os datasets a serem analizados. Cada subpasta aqui deve ser um único dataset. A execução do script run_ceq.py lê essa pasta e os arquivos contidos nela para produzir resultados, assim os resultados estão em gmag_filtros/data/dataset_name/output/.

## Como usar o run_ceq.py

Antes de qualquer coisa, garanta que o terminal esteja aberto (e com ambiente virtual ativado) na pasta onde está o run_ceq.py, ou seja, no terminal deve ter um caminho desse tipo: **seu_caminho/gmag_filtros/code/equivalent_layer**.

Cada dataset deve ser salvo dentro da pasta gmag_filtros/data/**dataset_name**. Nessa pasta deve conter um arquivo com os dados grav e outro com os dados mag. Cada arquivo de dados observados deve seguir a ordem: X,Y,Z,Obs, onde X é Easting, Y é Northing, Z é upwards e Obs é o dado observado. Lembrando que o sistema de coordenadas esperado para a execução da Camada Equivalente Clássica (script run_ceq.py) é o ENU, onde Eixo X é Easting, Y é Northing e Z é Upwards em metros, positivo para cima.

O arquivo pipeline_config_template.json presente na raiz do diretório é o modelo que deve ser usado para preencher as informações necessárias para construção da camada e transformações de campo. 
Se você usar o script de criação de dado sintético, um arquivo chamado **pipeline_config.json** já terá sido criado na sua pasta, caso contrário, copie e cole na sua pasta de dados e o preencha com as informações necessárias.

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
        "reg_min": 1e-6,
        "reg_max": 1e2,
        "steps": 20
    },


    "magnetic_transformations": {
        "upward_continuation": {
            "execute": true,
            "dz": 500
        },
        "reduction_to_pole": {
            "execute": true,
            "inc_new": 90,
            "dec_new": 0
        }
    },
    "gravity_transformations": {
        "upward_continuation": {
            "execute": false,
            "dz": 200
        }
    },
    "joint_analysis": {
        "poisson_ratio": {
            "execute": true,
            "density_threshold": 0.05,
            "mu0_min": 1e-6,
            "mu0_max": 1e4,
            "steps": 30
        }
    }
}
```

Para executar a construção da Camada Equivalente Clássica (um prisma embaixo de cada ponto de observação), vá na pasta gmag_filtros/code/equivalent_layer/ e encontre o arquivo run_ceq.py, esse é o arquivo que executa esta tarefa. Então, no terminal digite o seguinte comando:

```shell
python run_ceq.py --dataset DATASET_NAME --type DATA_TYPE
```

Onde DATASET_NAME é o nome da sua pasta de dados e DATA_TYPE é o tipo de dado que será processado (grav/gravity ou mag/magnetic). Dê Enter e espera a finalização do processo, uma mensagem aparecerá no terminal quando o processo estiver completo.

## A fazer

- Implementar as transformações: Derivadas, HGM, Tilt, ASA e Deconvolução de Euler

- Para o caso da RTP, adicionar o vinculo de positividade para garantir magnetização estimada positiva, isso necessita mudar a forma de resolver o sistema já que Cholesky não será possível nessa situação

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

## Como usar

Cada dataset deve ser salvo dentro da pasta project/data/"dataset_name". Nessa pasta deve conter um arquivo com os dados grav e outro com os dados mag (X,Y,Z,Obs). O arquivo metadata_template.json presente na raiz do diretório é o modelo que deve ser usado para preencher as informações necessárias para construção da camada e transformações de campo. Copie esse arquivo na pasta do dataset e coloque as informações necessárias.

```json
{
    "project_settings": {
        "data_type": "magnetic",
        "filename": "model1_mag.csv",
        "output_folder": "results_model1"
    },
    "data_columns": {
        "easting_col": "X",
        "northing_col": "Y",
        "upwards_col": "Z",
        "field": "obs"
    },
    "inversion_parameters": {
        "z_top": -300,
        "z_bottom": -800,
        "incf": 90,
        "decf": 0,
        "incs": 90,
        "decs": 0
    },
    "transformations": {
        "upward_continuation": {
            "execute": true,
            "dz": 500
        },
        "downward_continuation": {
            "execute": true,
            "dz": -150
        },
        "reduction_to_pole": {
            "execute": true,
            "inc_new": 90,
            "dec_new": 0
        }
    }
}
```

Para executar a construção da Camada Equivalente Clássica (um prisma embaixo de cada ponto de observação), vá na pasta project/code/equivalent_layer/ e encontre o arquivo run_ceq.py, esse é o arquivo que executa esta tarefa. Abre o arquivo e procure a linha onde está definido **dataset_name** e coloque o nome da sua pasta dataset, como mostrado no exemplo abaixo, salve e execute o script, os resultados estarão em project/output/results_dataset:

```python
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

PROJECT_HOME = CODE_ROOT.parent
DATA_FOLDER = PROJECT_HOME / "data"
BASE_FOLDER = PROJECT_HOME / "output" 

import modules.utils as ut
import modules.kernel as kn

def main():
    # ---------------------------------------------------------
    # 1. Load Data & Configs
    # ---------------------------------------------------------
    dataset_name = "model2"  # WRITE HERE THE FOLDER NAME OF WHERE DATASET IS! 
                            # THEN RUN TO EXECUTE EQ. LAYER BUILDING AND FIELD TRANSFORMATIONS
    
    
    print("==== Start ====\n")
    # Read the single unified configuration file
    config = ut.find_read_json_file(DATA_FOLDER / dataset_name, "pipeline_config.json")
```

Lembrando que o sistema de coordenadas esperado para a execução da Camada Equivalente Clássica (script run_ceq.py) é o ENU, onde Eixo X é Easting, Y é Northing e Z é Upwards, positivo para cima.

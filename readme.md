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

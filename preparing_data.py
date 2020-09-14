import pandas as pd
import string
import torch
import glob
import unicodedata
import os
#Transformação de arquivo csv para txt
nomes = pd.read_csv("babynames_clean.csv")
nomes.head()

nomes = nomes.rename(columns={'john':'nome'})
nomes = nomes.rename(columns={'boy':'sexo'})

nomes["sexo"].unique()

selecao_meninas = nomes["sexo"] ==  "girl"
nome_meninas = nomes[selecao_meninas]
nome_meninas.head()

selecao_meninos = nomes["sexo"] ==  "boy"
nome_meninos = nomes[selecao_meninos]
nome_meninos.head()

with open("meninos.txt", 'w') as out:
    for n in list(nome_meninos['nome'].drop_duplicates()):
        out.write(n + "\n")

with open("meninas.txt", 'w') as out:
    for n in list(nome_meninas['nome'].drop_duplicates()):
        out.write(n + "\n")


#Preparacao dos dados

#Lista de nomes para cada categoria
nomes_categoria = {}
#Vetor com todas as categorias diferentes do arquivo
todas_categorias = []

#todas as letras da tabela ASCII (minusculas e maiusculas)
letras = string.ascii_letters + " .,;'"
#total de letras
n_letras = len(letras)
#Total de categorias identificadas




#retorna todos os caminhos identificados por um determinado padrão
def arquivos(caminho):
    return glob.glob(caminho)

    

#Transforma um caracter de UNICODE para o equivalente da tabela ASCII
#Fonte: https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'and c in letras)

   

#Método para ler todas as linhas de um arquivo e transformar cada uma delas em caracteres da tabela ASCII
def lerLinhas(arquivo):
    linhas = open(arquivo, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(linha) for linha in linhas]

#percorre todos os arquivos, um por vez e usa seu nome como categoria. Adiciona a mesma a lista de todas as categorias
#, lê e adiciona todos os nomes presentes naquela categoria (arquivo) a seu respectivo objeto 
# (Ex.: nomes_categoria[meninos] += Carlos )
for arquivo in arquivos('*.txt'):
    categoria = os.path.splitext(os.path.basename(arquivo))[0]
    todas_categorias.append(categoria)
    linhas = lerLinhas(arquivo)
    nomes_categoria[categoria] = linhas

    


n_categorias = len(todas_categorias)


#Transformar nomes em tensores
    

#Encontra o índice da letra indicada, onde o vetor de letras se dá primeiramente por todas as letras maiusculas
#e depois todas as letras minusculas
def letterToIndex(letter):
    return letras.find(letter)

#Demonstração de transformação de uma letra em um tensor (um vetor nesse caso)
def letterToTensor(letter):
    #preenche o tensor com zeros em todas posições, sendo que o total de posicoes é igual ao total de letras 
    #da tabela ASCII.
    tensor = torch.zeros(1, n_letras)
    #Para a posição da letra desejada, o valor se torna 1 ao invés de 0
    tensor[0][letterToIndex(letter)] = 1
    return tensor

#Transforma uma linha em tensores, onde cada letra do nome será um tensor separado. É um método
#mais eficaz e rápido do que o método anterior
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letras)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


from __future__ import unicode_literals, print_function, division
import pandas as pd
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import math
import time
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

#retorna todos os caminhos identificados por um determinado padrão
def arquivos(caminho):
    return glob.glob(caminho)

#todas as letras da tabela ASCII (minusculas e maiusculas)
letras = string.ascii_letters + " .,;'"
#total de letras
n_letras = len(letras)

#Transforma um caracter de UNICODE para o equivalente da tabela ASCII
#Fonte: https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'and c in letras)

#Lista de nomes para cada categoria
nomes_categoria = {}
#Vetor com todas as categorias diferentes do arquivo
todas_categorias = []

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

#Total de categorias identificadas
n_categorias = len(todas_categorias)

#Para cada uma delas, imprime seu identificador
for categoria in todas_categorias:
    print("Categoria: " + categoria)

#Imprime o total de categorias
print("Total de categorias: " + str(n_categorias))

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

#Transforma uma linha em um tensor onde o tensor se dá por [tamanho_da_linha, 1, total_letras]
#Nesse caso o tensor é uma matriz e para cada letra da linha o valor é substituido de 0 para 1.
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letras)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print("Tensor de uma única letra" + letterToTensor('t'))
print("Tensor de uma linha" + lineToTensor('Gustavo').size())


#Classe da rede neural recorrente
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        #?????????
        self.hidden_size = hidden_size

        #i2h é a camada de rede onde o estado oculto será gerenciado, sendo atualizado a cada iteração
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        #i2o é a camada de rede onde o resultado será gerenciado
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        #????????
        self.softmax = nn.LogSoftmax(dim=1)

    #Método que faz com que a rede executa uma iteração
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letras, n_hidden, n_categorias)

#exemplo
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)

#Treinamento
#Taxa de aprendizado
learning_rate = 0.005 
criterion = nn.NLLLoss()

#Classe de treinamento
def train(category_tensor, line_tensor):
    #Estado inicial da rede
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()



#execucao do treinamento
#Numero de iterações da rede
n_iters = 100000
#A cada x iterações será exibida uma das iterações feitas pela rede, mostrando o tempo que levou até ser executada,
#a probabilidade dada pela rede e usada para identificar a categoria, o nome avaliado, a categoria identificada,
#se está correto ou incorreto. No caso de incorreto, mostra a categoria que seria correta entre parenteses.
print_every = 5000
#A cada x iterações, o número de perdas será armazenado para ser mostrado no gráfico.
plot_every = 1000


#Treinando a rede aleatoriamente
#Recebe uma lista e retorna um valor aleatório
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

#Método que irá efetivamente realizar o treino
def randomTrainingExample():
    #Recebe uma categoria aleátoria
    category = randomChoice(todas_categorias)
    #Recebe uma linha aleatória
    line = randomChoice(nomes_categoria[category])
    #Transforma a categoria em um tensor.
    category_tensor = torch.tensor([todas_categorias.index(category)], dtype=torch.long)
    #Transforma a linha em um tensor
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


#interpretar o resultado da categoria
#Recebe o resultado como parametro e retorna a categoria a qual o nome testado pertence (em índice e textualmente)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return todas_categorias[category_i], category_i

#Variável de controle das perdas, indica quão mal a interpretação de um valor foi
current_loss = 0
all_losses = []

#funcao que exibe o tempo decorrido até determinado momento
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    #print(categoryFromOutput(output))
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#for i in range(10):
  #  category, line, category_tensor, line_tensor = randomTrainingExample()
  #  print('category =', category, '/ line =', line)

#Resultados
plt.figure()
plt.plot(all_losses)

#matriz de confusao
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categorias, n_categorias)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = todas_categorias.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categorias):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + todas_categorias, rotation=90)
ax.set_yticklabels([''] + todas_categorias)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

sphinx_gallery_thumbnail_number = 2
plt.show()

#input do usuario
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        # Get top N categories
        topv, topi = output.topk(n_predictions-1, 1, True)
        predictions = []

        for i in range(n_predictions-1):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, todas_categorias[category_index]))
            predictions.append([value, todas_categorias[category_index]])

predict('Luciano')
predict('Maria')
predict('Gustavo')
predict('Joana')
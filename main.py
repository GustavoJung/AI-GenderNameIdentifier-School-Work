from __future__ import unicode_literals, print_function, division
from preparing_data import *
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
import PySimpleGUI as sg



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


#Treinamento
#Taxa de aprendizado
learning_rate = 0.005 
criterion = nn.NLLLoss()


if __name__ == '__main__':
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
    n_iters = 10000
    #A cada x iterações será exibida uma das iterações feitas pela rede, mostrando o tempo que levou até ser executada,
    #a probabilidade dada pela rede e usada para identificar a categoria, o nome avaliado, a categoria identificada,
    #se está correto ou incorreto. No caso de incorreto, mostra a categoria que seria correta entre parenteses.
    print_every = 500
    #A cada x iterações, o número de perdas será armazenado para ser mostrado no gráfico.
    plot_every = 100


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

    corrects = 0
    #output
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        #acurácia
        a_guess,a_guess_i=categoryFromOutput(output)
        if a_guess == category:
                corrects = corrects+1

        # Print iter number, loss, name and guess
        #print(categoryFromOutput(output))
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = 'V' if guess == category else 'X (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    #imprime a acuracia
    print("Acurácia")
    print(corrects/(n_iters))
    

    #for i in range(10):
    #  category, line, category_tensor, line_tensor = randomTrainingExample()
    #  print('category =', category, '/ line =', line)

    #Resultados
    plt.figure()
    plt.plot(all_losses)

    torch.save(rnn.state_dict(), "rede_treinada.pth")

    #matriz de confusao
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categorias, n_categorias)
    n_confusion = 1000

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
        #print(category_i)
        #print(guess)

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



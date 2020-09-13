import torch
import string
from preparing_data import *
from main import RNN
import PySimpleGUI as sg

model = RNN(n_letras,128,len(todas_categorias))
#model = torch.load("rede_treinada.pth")
model.load_state_dict(torch.load("rede_treinada.pth"))
model.eval()

#Total de categorias identificadas
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

# Just return an output given a line
def evaluate(line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

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
    return predictions

layout = [
    [sg.Text("Nome a ser previsto pela rede:  "),sg.Input(key='__in__')],
    [sg.Button("Prever"), sg.Button("Sair")]
 ]

window = sg.Window('Previsão de gênero de um nome!',layout, size=(400,200))

while True:
    event, values = window.read()
    if event == 'Sair' or event == sg.WIN_CLOSED:
        break
    if event=='Prever':
        higher_predition = predict(values['__in__'])[0]
        lower_predition = predict(values['__in__'])[1]
        result_prob = higher_predition[0]
        result_prob_not = lower_predition[0]
        result_prevision = higher_predition[1]
        sg.Popup("Gênero do nome: " + result_prevision + " \nprobabilidade calculada de ser: " 
        + str(round(result_prob,2)) + "\nprobabilidade calculada de não ser: " +  str(round(result_prob_not,2)))

window.close()
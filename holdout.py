from preparing_data import *
import random

meninos = []
meninas = []

for arquivo in arquivos('*.txt'):
    categoria = os.path.splitext(os.path.basename(arquivo))[0]
    if categoria == 'meninos':
        linhas = lerLinhas(arquivo)
        meninos.append(linhas)
    linhas = lerLinhas(arquivo)
    meninas.append(linhas)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

sub_testes = []
sub_treinamento = []
every = 50

#200 meninos testes
for iter in range(1, (10001)):
     if iter % every == 0:
        sub_testes.append(randomChoice(nomes_categoria['meninos']))

#200 meninas testes 
for iter in range(1, (10001)):
     if iter % every == 0:
        sub_testes.append(randomChoice(nomes_categoria['meninas']))

#200 meninos treinamento
for iter in range(1, (10001)):
     if iter % every == 0:
        sub_treinamento.append(randomChoice(nomes_categoria['meninos']))
    
#200 meninas treinamento
for iter in range(1, (10001)):
     if iter % every == 0:
        sub_treinamento.append(randomChoice(nomes_categoria['meninas']))
    
print((len(sub_testes)))
print((len(sub_treinamento)))
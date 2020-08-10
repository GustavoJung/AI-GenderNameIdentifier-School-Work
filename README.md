# inteligenciaArtificial

## Equipe
Gustavo Jung e Luciano Velho Garcia

## Problema
Identificar o gênero (masculino ou feminino) de alguém pelo seu primeiro nome

## Dataset 
https://data.world/alexandra/baby-names onde o mesmo foi desenvolvido pelo usuário Alexandra, link para o perfil: https://data.world/alexandra. O Dataset está em formato .csv, tem 7000 itens e possuí duas colunas: nome e gênero, ambas em formato de String.

## Técnica 
  Será desenvolvia uma RNN (Recurrent Neural Network) onde primeiramente serão adaptados os nomes e transformados em tensores. Primeiro, os nomes serão transformados de Unicode para ASCII, utilizando a biblioteca Unidecode (link: https://pypi.org/project/Unidecode/). Feito isso, os nomes devem ser transformados em tensores, para transformá-los serão criados vetores de tamanho igual ao número de letras(maiúsculas e minúsculas) da tabela ASCII, onde cada índice terá valor 0, inicialmente. Para cada letra do nome, será identificado sua posição nas letras da tabela ASCII (por exemplo: a letra c teria como posição, 3. 

Exemplo de um vetor da letra c imaginando que teríamos apenas as 26 primeiras letras minúsculas vindas da tabela ASCII:

tensor = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


  A rede é estruturada com um input inicial(letra atual de um nome transformado em tensor) e um estado oculto inicial (zeros), as duas camadas serão combinadas e um novo input(próxima letra do nome) é combinado com o estado inicial armazenado da operação anterior, o resultado da combinação é dividido em duas camadas(iguais), uma para o fluxo da rede e para o resultado final e outra para gerar novos estados ocultos que fomentarão a própria rede. A camada de fluxo levará à próxima camada que á de ativação. Cada etapa gera um output, que são as probabilidades do nome ser do gênero masculino ou feminino, quanto maior o valor maior a probabilidade. O output final acontece quando chega-se ao fim das letras do nome.

## Treinamento da rede 
  O treinamento ocorrerá da seguinte forma: criar um input e marcar um tensor alvo, o estado inicial da camada oculta é zerado,  para cada letra do nome: ler e salvar seu estado oculto para a próxima letra, comparar o resultado com o alvo marcado, fazer uma backpropagation e retornar o resultado com os erros.

## Avaliação dos resultados
  Para verificar o desempenho da rede serão desenvolvidas matrizes de confusão seguindo dois modelos: Holdout e K-Fold.

#### Holdout
  O Dataset será dividido em dois (proporcionalmente), um subconjunto será de testes e outro de treinamento. A partir dos treinamentos feitos com esse respectivo subconjunto, será testado com o subconjunto de teste e com esse resultado, montada a matriz de confusão.

#### K-Fold
  O Dataset será dividido em K subconjuntos (proporcionalmente), para cada um deles existirão os subconjuntos de treinamento e testes. Serão realizadas K rodadas de treinamento e para cada uma, testes. O resultado será avaliado a partir da média de cada uma das rodadas e, a partir dele, montada a matriz de confusão. 

# inteligenciaArtificial

• Equipe: Gustavo Jung e Luciano Velho Garcia
• Problema: Identificar o gênero (masculino ou feminino) de alguém pelo seu primeiro nome
• Dataset: https://data.world/alexandra/baby-names
• Técnica: descrever brevemente como a técnica de Inteligência Computacional será aplicada
(ou seja, como o problema será modelado para aplicação da técnica).
Será desenvolvia uma RNN (Recurrent Neural Network) onde primeiramente serão adaptados os nomes e transformados em tensores. A rede é estruturada com um input inicial e um estado oculto inicial, após sua combinação um novo input é combinado com o estado inicial armazenado da operação anterior e, ao fim, um output final. O treinamento ocorrerá da seguinte forma: criar um input e marcar um tensor alvo, o estado inicial da camada oculta é zerado,  para cada letra do nome: ler e salvar seu estado oculto para a próxima letra, comparar o resultado com o alvo marcado, fazer uma backpropagation e retornar o resultado com os erros.


O usuário poderá informar nomes para teste.

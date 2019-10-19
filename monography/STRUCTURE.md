# Introdução

## Motivação

### Científica

>  

### Social

> Controle de congestionamento é uma parte essencial para cidades inteligentes
> Previsão de tráfego não só ajuda os motoristas como também pode ajudar no controle de semáforos da cidade
> Trafego mais seguro
> Menos poluição
> Menos perda de tempo por parte dos motoristas
> Menor gastos com gasolina

## Área

> As metodologias utilizadas na área de previsão podem ser divididas em duas categorias
> Existem as metodologias paramêtricas e não-paramêtricas
> Modelos paramêtricos possuem um número fixo de parametros e normalmente são computacionalmente menos exigentes
> Modelos não paramêtricos são mais flexíveis quanto a número de parâmetros, porém mais exigentes computacionalmente
> Aprendizagem de máquina superam as heurísticas
> Metodologias mais usada na aprendizagem de máquina é LSTM e variações
> Metodologias normalmente usam de dados de vias livres

## Hipótese

> Usar a metodologia da LSTM também serve para previsão em vias semafóricas 

## Justificativa

> Não existe, até a escrita desse artigo, uma previsão em vias semafóricas.
> Metodologia que funcionem para vias semafóricas são necessárias, visto que a maioria das vias de uma cidade possuem algum tipo de obstrução, leia como não-livres

PS: conferir essa minha afirmação

## Objetivo

> Verificar a qualidade da metodologia de aprendizagem de máquina usando LSTM consegue prever satisfatoriamente o fluxo de vias semafóricas

## Limitações

> Acidentes influênciam
> Dados não de poucos meses
> Redes neurais não são perfeitas

## Organização do trabalho

## Resultados Esperados

## Metodologia

> Usaremos os dados providos do detran
> Treinaremos a rede neural

# Fundamentação Teorica

## Redes Neurais

> Redes neurais é top
> As redes neurais possuem limitações quanto a se lembrar de coisas passadas

## Redes Neurais Recorrentes

> Redes Neurais recorrentes é um avanço em cima das redes neurais
> Elas conseguem se lembrar
> Utilizam de gates
> Possuem dois defeitos

## LSTM

> Melhora da Redes neurais recorrentes para consertar os defeitos dela
> Possui ainda um defeito

# Trabalhos Relacionados
> Trabalho do SAE é doido
> Trabalho de LSTM em Seoul
> Trabalho de LSTM 2
> Trabalho de LSTM 3
> Trabalho do GRU?

- Tabela que mostra a diferença entre o meu trabalho e o dos outros

# Metodologia

- Resumo Introdução
- Modelagem do problema
- Coleta de Dados
- Pré-Processamento dos dados
	- Quais são as características dos dados
		- Stocastico?
		- Não linear?
		- Dependente do tempo e espaço
	- Quais atributos eu tenho
	- Quais atributos eu vou retirar e porquê
	- Quais transformações vou fazer
	- Quais dados foram normalizados e de que forma
	- Quais dados foram gerado a partir dos dados originais
		- Falar qual é o padrão para quebrar data para redes que seja mais fácil para redes reurais
			- Não pode colocar como numérico, pois acaba se criando uma relação dos números (quarta é maior que segunda)
- Implementação
	- Justificar o uso de ReLu
	- Justificar o tamanho do output escolhido
	- Justificar quantas camadas eu escolhi
	- Justificar pq LSTM simples e n outras versões
	- Justificar pq usar RNN
- Uso
	- Explicar como foi o treinamento
- Avaliação dos resultados
	- Explicar metodologia (rmse)
	- Falar sobre o k-fold para séries temporais
	- Explicar o pq de usar essa metodologia

# Resultados

- Aprofundamento da metodologia usada
- Mostra dos resultados (Gráficos)
	- Colocar métricas de tempo de treinamento
	- Colocar métricas de tempo de previsão
	- Quantas epocas demorou cada modelo para divergir
	- Analise de performance em fluxo com baixo tráfego
	- Analise de performance em fluxo com medio tráfego
	- Analise de performance em fluxo com alto tráfego
	- Grid Search (mostra como as variações de parametros influenciam no resultado)
		- Mostrar como os resultados variam se eu tirar weekday
		- Mostrar como os resultados variam se eu colocar velocidade média no intervalo
- Explicação dos resultados

# Conclusão

- Resumo do resultado
- Verificação da Hipótese
- Falhas no trabalho
- Trabalhos futuros

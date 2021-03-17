# Programa: demanda.py
# Authors: Ramon R. Valeriano e Chrislaine B. Marinho
# Description: Algorimto para Calculos do emplacamento dos carros.
# Developed: 13/11/2020 - 13:41 Revisão 0
# Updated:   16/11/2020 - 15:41 Revisão 1
# Updated:   17/11/2020 - 18:03 Revisão 2
# Updated:   18/11/2020 - 10:12 Revisão 3
# Updated:   17/03/2020 - 11:02 Revisão 4

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


"""-----------------------------Coletando os cados-----------------------------------------"""
# ---------------------------------Modelo IX35------------------------------------------------
nome1 = 'IX35'
file1 = 'ix35.csv'
coletando1 = pd.read_csv(file1, delimiter=';')
coletando1.Data = pd.to_datetime(coletando1.Data)
print(coletando1.head())

# -------------------------------=Modelo TUCSON-----------------------------------------------
nome2 = 'Tucson'
file2 = 'Tucson.csv'
coletando2 = pd.read_csv(file2, delimiter=';')
coletando2.Data = pd.to_datetime(coletando2.Data)
print(coletando2.head())

# Gráfico para mostrar se os dados estão funcionando.
"""sns.lineplot(x='Data', y='Emplacamentos', data=coletando4, label='Dados que serão usados')
ax = plt.xticks(rotation=70)
plt.show()"""


"""-------Fazendo com que o norro programa mantenha o nosso padrão, igual para todos-------"""
# ---------------------------------Modelo IX35------------------------------------------------
sc = StandardScaler()
sc.fit(coletando1['Soma'].values.reshape(-1, 1))
y = sc.transform(coletando1['Soma'].values.reshape(-1, 1))

tamanho_treino = int(len(coletando1)*0.9)
tamanho_teste = int(len(coletando1)-tamanho_treino)

ytreino = y[0:tamanho_treino]
yteste = y[tamanho_treino:len(coletando1)]

# -------------------------------=Modelo TUCSON-----------------------------------------------
sc1 = StandardScaler()
sc1.fit(coletando2['Soma'].values.reshape(-1, 1))
y1 = sc.transform(coletando2['Soma'].values.reshape(-1, 1))

tamanho_treino1 = int(len(coletando2)*0.9)
tamanho_teste1 = int(len(coletando2)-tamanho_treino1)

ytreino1 = y1[0:tamanho_treino1]
yteste1 = y1[tamanho_treino1:len(coletando2)]


# Gráfico para mostrar se os dados estão funcionando.
"""sns.lineplot(x='Data', y=ytreino3[:,0], data=coletando4[0:tamanho_treino3], label='Treino')
sns.lineplot(x='Data', y=yteste3[:,0], data=coletando4[tamanho_treino3:len(coletando4)], label='Teste')
ax = plt.xticks(rotation=70)
plt.show()"""

"""Ajustando dados para o treino."""

"""----------------Função para separar os dados--------------------------------------------"""
def separa_dados(vetor,n_passos):
    """Entrada: vetor: número de passageiros
               n_passos: número de passos no regressor
     Saída:
              X_novo: Array 2D
              y_novo: Array 1D - Nosso alvo
    """
    X_novo, y_novo = [], []
    for i in range(n_passos,vetor.shape[0]):
        X_novo.append(list(vetor.loc[i-n_passos:i-1]))
        y_novo.append(vetor.loc[i])
    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo

# ---------------------------------Modelo IX35------------------------------------------------
vetor = pd.DataFrame(ytreino)[0]
vetor2 = pd.DataFrame(yteste)[0]

xtreino_novo, ytreino_novo = separa_dados(vetor, 1)
xteste_novo, yteste_novo = separa_dados(vetor2, 1)

xtreino_novo = xtreino_novo.reshape((xtreino_novo.shape[0],xtreino_novo.shape[1],1))
xteste_novo = xteste_novo.reshape((xteste_novo.shape[0],xteste_novo.shape[1],1))

# -------------------------------=Modelo TUCSON-----------------------------------------------
vetor_1 = pd.DataFrame(ytreino1)[0]
vetor2_1 = pd.DataFrame(yteste1)[0]

xtreino_novo1, ytreino_novo1 = separa_dados(vetor_1, 1)
xteste_novo1, yteste_novo1 = separa_dados(vetor2_1, 1)

xtreino_novo1 = xtreino_novo1.reshape((xtreino_novo1.shape[0],xtreino_novo1.shape[1],1))
xteste_novo1 = xteste_novo1.reshape((xteste_novo1.shape[0],xteste_novo1.shape[1],1))

"""---------------------------Montando as Redes Neurais------------------------------------"""
# -------Será usada como uma variável global para ser usadas em todas, estou usando 900------.

EPOCAS = 5

# ---------------------------------Modelo IX35------------------------------------------------
recorrente = Sequential()

recorrente.add(LSTM(128, input_shape=(xtreino_novo.shape[1],xtreino_novo.shape[2])
                    ))
recorrente.add(Dense(units=1))

recorrente.compile(loss='mean_squared_error',optimizer='RMSProp')

resultado = recorrente.fit(xtreino_novo,ytreino_novo,
                              validation_data=(xteste_novo,yteste_novo),epochs=EPOCAS)

y_ajustado = recorrente.predict(xtreino_novo)
y_predito = recorrente.predict(xteste_novo)

# -------------------------------=Modelo TUCSON-----------------------------------------------
recorrente1 = Sequential()

recorrente1.add(LSTM(128, input_shape=(xtreino_novo1.shape[1],xtreino_novo1.shape[2])
                    ))
recorrente1.add(Dense(units=1))

recorrente1.compile(loss='mean_squared_error',optimizer='RMSProp')

resultado1 = recorrente1.fit(xtreino_novo1,ytreino_novo1,
                              validation_data=(xteste_novo1,yteste_novo1),epochs=EPOCAS)

y_ajustado1 = recorrente1.predict(xtreino_novo1)
y_predito1 = recorrente1.predict(xteste_novo1)

"""----------------Função para reconverter os dados fora da escala, da forma original.--------------------------"""

def conversao_apresentar(datas, y, nome1, nome2):
    lista_data = list(datas)
    resultadoy = sc.inverse_transform(y)
    d1 = {
        nome1: lista_data,
        nome2: resultadoy
    }
    transformando = pd.DataFrame(d1)
    transformando.columns = [nome1, nome2]
    return transformando

"""Fazendo o algoritmo para tirar os dados da escala e que o código apareça coerente ao plotar os dados na tela."""
"""--------------------PREPARANDO OS DADOS PARA MOSTRAGEM-----------------------------------"""

nome1 , nome2 = 'Data', 'Emplacamentos'  # Dados que serão geral para todos.

# ---------------------------------Modelo IX35------------------------------------------------
"""Quando se refere aos dados de treinamento. """
datas1 = list(coletando1.Data[0:tamanho_treino])
y_tr = ytreino[:,0].round(2)
transformando1 = conversao_apresentar(datas1, y_tr, nome1, nome2)
#print(transformando1)

"""Fazendo o algorimto para apresentar os dados do ajustado de forma coerente. """
datas2 = list(coletando1.Data[0:64])
y_ta = y_ajustado[:,0].round(2)
transformando2 = conversao_apresentar(datas2, y_ta, nome1, nome2)
#print(transformando2)

"""Fazendo o algoritmo para apresentar os dados do teste de forma coerente. """
datas3 = list(coletando1.Data[tamanho_treino:len(coletando1)])
y_ts = yteste[:,0]
transformando3 = conversao_apresentar(datas3, y_ts, nome1, nome2)
#print(transformando3)

"""Fazendo o algoritmo para apresentar os dados da previsão de forma coerente. """
datas4 = list(coletando1.Data[tamanho_treino+1:len(coletando1)])
y_pe = y_predito[:, 0]
transformando4 = conversao_apresentar(datas4, y_pe, nome1, nome2)
#print(transformando4)

# -------------------------------=Modelo TUCSON-----------------------------------------------
"""Quando se refere aos dados de treinamento. """
datas1_1 = list(coletando2.Data[0:tamanho_treino1])
y_tr1 = ytreino1[:,0].round(2)
transformando1_1 = conversao_apresentar(datas1_1, y_tr1, nome1, nome2)
#print(transformando1_1)

"""Fazendo o algorimto para apresentar os dados do ajustado de forma coerente. """
datas2_1 = list(coletando2.Data[0:64])
y_ta1 = y_ajustado1[:,0].round(2)
transformando2_1 = conversao_apresentar(datas2_1, y_ta1, nome1, nome2)
#print(transformando2_1)

"""Fazendo o algoritmo para apresentar os dados do teste de forma coerente. """
datas3_1 = list(coletando2.Data[tamanho_treino1:len(coletando2)])
y_ts1 = yteste1[:,0]
transformando3_1 = conversao_apresentar(datas3_1, y_ts1, nome1, nome2)
#print(transformando3)

"""Fazendo o algoritmo para apresentar os dados da previsão de forma coerente. """
datas4_1 = list(coletando2.Data[tamanho_treino1+1:len(coletando2)])
y_pe1 = y_predito1[:, 0]
transformando4_1 = conversao_apresentar(datas4_1, y_pe1, nome1, nome2)
#print(transformando4_1)

"""------Area responsável por montar todos os gráficos do nosso sistema de previsão.-------"""

"""---------------Area responsável por montar todos os gráficos do juntos.-----------------"""

plt.figure(figsize=(22, 20))
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = plt.subplot(4, 1, 1)
ax.set_title('Demanda', fontsize = 18, loc='left')
sns.lineplot(x = nome1, y = nome2, data = transformando1, label = 'Treino IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando2, label = 'Ajuste Treino IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando3, label = 'Teste IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando4, label = 'Previsão IX35')
#plt.xticks(rotation = 70)
#plt.figure(figsize=(16, 12))
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = plt.subplot(4, 1, 2)
sns.lineplot(x = nome1, y = nome2, data = transformando1_1, label = 'Treino TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando2_1, label = 'Ajuste Treino TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando3_1, label = 'Teste TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando4_1, label = 'Previsão TUSCON')
#plt.xticks(rotation = 70)
#plt.figure(figsize=(16, 12))
plt.show()

"""-----------------------Mostrar de Forma Individual.-------------------------------------"""
# ---------------------------------Modelo IX35------------------------------------------------
sns.lineplot(x = nome1, y = nome2, data = transformando1, label = 'Treino IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando2, label = 'Ajuste Treino IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando3, label = 'Teste IX35')
sns.lineplot(x = nome1, y = nome2, data = transformando4, label = 'Previsão IX35')
plt.xticks(rotation = 70)
plt.show()

# -------------------------------=Modelo TUCSON-----------------------------------------------
sns.lineplot(x = nome1, y = nome2, data = transformando1_1, label = 'Treino TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando2_1, label = 'Ajuste Treino TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando3_1, label = 'Teste TUCSON')
sns.lineplot(x = nome1, y = nome2, data = transformando4_1, label = 'Previsão TUSCON')
plt.xticks(rotation = 70)
plt.show()

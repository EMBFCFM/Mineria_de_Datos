# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#cargamos los datos de entrada
data = pd.read_csv("./Celulares.csv")
#veamos cuantas dimensiones y registros contiene
data.shape

#son 1000 registros con 8 columnas. Veamos los primeros registros
data.head()

# Visualizamos rápidamente las caraterísticas de entrada
data.drop(['Date_Start','id','blue','four_g','three_g','touch_screen','wifi','dual_sim'],1).hist()
plt.show()

# Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
# esto es en el eje X: entre 500 y 2000
# y en el eje Y: entre 0 y 4000
filtered_data = data[(data['battery_power'] <= 2000) & (data['ram'] <= 4000)]

colores=['orange','blue']
tamanios=[30,60]

f1 = filtered_data['battery_power'].values
f2 = filtered_data['ram'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de Cantidad de Palabras
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['battery_power']>1000):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()

# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =filtered_data[["battery_power"]]
X_train = np.array(dataX)
y_train = filtered_data['ram'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coeficientes: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Termino independiente: \n', regr.intercept_)
# Error Cuadrado Medio
print("Error medio cuadrado: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Valor de varianza: %.2f' % r2_score(y_train, y_pred))

#Vamos a comprobar:
# Quiero predecir cuántos "Shares" voy a obtener por un artículo con 2.000 palabras,
# según nuestro modelo, hacemos:
y_Dosmil = regr.predict([[3000]])
print(int(y_Dosmil))
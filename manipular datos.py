import pandas as pd
datos=pd.read_csv('Celulares.csv',header=0)
"""Imprime la base de datos"""
print(datos)
"""Imprime la columna int_memory"""
print(datos['int_memory'])

"""Imprime la columna battery_power en orden descendente"""
print(datos.sort_values(by='battery_power', ascending=False))

"""Imprime los datos de la columna battery_power menores a 550"""
bateria=datos['battery_power']
print("Baterias")
print(bateria[bateria<550])

print("Media en la bateria")
media=datos.battery_power.median()
print(media)
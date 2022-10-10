import pandas as pd

df=pd.read_csv("Celulares.csv",header=0)
"""Remplaza el valor de la fila 5 de la columna fc"""
df.loc[5, 'fc']='BBB'

"""Elimina la columna four_g"""
df.pop('four_g')

"""Elimina la columna m_dep"""
df.pop('m_dep')

""""Elimina la columna int_memory"""
df.pop('int_memory')

df.to_csv("cambios.csv",index=False)

print(df)


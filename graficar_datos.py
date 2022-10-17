import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

Data =pd.read_csv("Celulares.csv", usecols=['type_device','battery_power'])
df=DataFrame(Data, columns=['type_device','battery_power'])
df.plot(x='type_device',y='battery_power', kind='line')
plt.show()



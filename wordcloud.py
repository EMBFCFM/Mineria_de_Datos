#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud
#Importing Dataset
df = pd.read_csv("Celulares.csv")
#Checking the Data
df.head()
#Creating the text variable
text = " ".join(type_define.split()[0] for type_define in df.type_device)
# Creating word_cloud with text as argument in .generate() method
wordcloud = WordCloud(collocations = False, background_color="white").generate(text)
# Display the generated Word Cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

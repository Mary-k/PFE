import pandas as pd  
import numpy as np

cols=["Post","Polarity Class"]
df = pd.read_csv('/home/mimi/Downloads/grid-export.csv',header=None)

#print(df.head())
df1 = pd.DataFrame(df,columns=[2,12]) 
print(df1.head(100))

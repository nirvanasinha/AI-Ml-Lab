import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
ds=pd.read_csv('C:/Users/kumar/Desktop/Datasets/2.csv')
x=ds.iloc[:,:].values
imp = Imputer(missing_values=np.nan, strategy='mean')
imp=imp.fit(x[:,-3:])
x=imp.transform(x[:,-3:])
print(x)
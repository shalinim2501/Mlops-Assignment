import pandas as pd
from sklearn.linear_model import LinearRegression
d=pd.read_csv('data1.csv')
x=d[['years']]
y=d['salary']
y=y*10
model=LinearRegression()
import joblib
joblib.dump(model, 'model.pkl')
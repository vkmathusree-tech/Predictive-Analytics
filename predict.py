import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('sales_data.csv')

X = data[['Month']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[9]])

print("Predicted Sales for Month 9:", prediction[0])

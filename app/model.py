import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle


dataset = pd.read_csv("app/biodeg_cleaned.csv")

y = dataset.issue
X = dataset.iloc[:,:-1]
X_train, X_test, y_train , y_test = train_test_split(X,y ,random_state=42 , test_size = 0.3)
print(type(X_train))


poly = PolynomialFeatures(2,include_bias = False)
poly.fit(X_train)
X_train_poly = (pd.DataFrame(poly.transform(X_train),columns = poly.get_feature_names(X_train.columns)))
X_test_poly = (pd.DataFrame(poly.transform(X_test),columns = poly.get_feature_names(X_train.columns)))


scaler_poly = StandardScaler()
scaler_poly.fit(X_train_poly)
X_train_poly = pd.DataFrame(scaler_poly.transform(X_train_poly),columns = X_train_poly.columns)
X_test_poly = pd.DataFrame(scaler_poly.transform(X_test_poly),columns = X_train_poly.columns)






Linear_regression_model = LogisticRegression(solver = "liblinear", penalty = "l2", C = 0.9, random_state=42)
Linear_regression_model.fit(X_train,y_train)



pickle.dump(Linear_regression_model,open("best_model.pkl" ,"wb"))



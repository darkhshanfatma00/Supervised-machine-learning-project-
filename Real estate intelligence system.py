import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from xgboost import XGBRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error,accuracy_score,precision_score,f1_score,confusion_matrix,recall_score
from sklearn.model_selection import train_test_split

data=pd.read_csv("data_set.txt")

# data  inception 

No_rows_columns=data.shape
print(No_rows_columns)
first_rows=data.head(5)
print(first_rows)
last_rows=data.tail(5)
print(last_rows)
count_nullVlaues=data.isna().sum()
print(count_nullVlaues)
count_nonNullValues=data.notna().sum()
print(count_nonNullValues)
datatypes=data.dtypes
print(datatypes)
statistics_info=data.describe()
print(statistics_info)
print(data.info())


# regression
X1 = data[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
Y1 = data["price"]
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=42)

model_regressor={
    "Linear regression":LinearRegression(),
    "Random forest regression":RandomForestRegressor(n_estimators=10,min_samples_leaf=2,min_samples_split=4,random_state=22),
    "Gradient boosting regressor":GradientBoostingRegressor(n_estimators=10,min_samples_leaf=2,min_samples_split=4,random_state=22),
    "Decision tree regressor":DecisionTreeRegressor( criterion="squared_error",max_depth=5,min_samples_leaf=1,min_samples_split=2),
    "XGB Regressor":XGBRegressor(n_estimators=10,random_state=42, objective="reg:squarederror"),
}

# Regression 
for name1 , model1 in model_regressor.items():
     print(name1,model1)
     pip_regressor=Pipeline([
          ("scaler",StandardScaler()),
          ("model1",model1),
     ])
     pip_regressor.fit(X1_train,Y1_train)
     prediction_regressor=pip_regressor.predict(X1_test)
     print(prediction_regressor)

     # evaluation
     mse=mean_squared_error(Y1_test,prediction_regressor)
     print(mse)
     mae=mean_absolute_error(Y1_test,prediction_regressor)
     print(mae)
     rmse=root_mean_squared_error(Y1_test,prediction_regressor)
     print(rmse)
     r2=r2_score(Y1_test,prediction_regressor)
     print(r2)


# classification 
model_classifier={
    "Logistics regression":LogisticRegression(),
    "Gradient Boosting classifier":GradientBoostingClassifier(n_estimators=10,min_samples_leaf=2,min_samples_split=4,random_state=22),
    "Random forest classifier":RandomForestClassifier(n_estimators=10,min_samples_leaf=2,min_samples_split=4,random_state=22 ),
    "Decision tree classfier":DecisionTreeClassifier( criterion="gini",max_depth=5,min_samples_leaf=1,min_samples_split=2),
    "XGB Classifier":XGBClassifier(n_estimators=10,random_state=22,  objective="binary:logistic"),
}

X2 = data[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
Y2 = data["is_expensive"]
X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.2,random_state=42)


for name2 , model2 in model_classifier.items():
     print(name2,model2)
     pip_classifier=Pipeline([
          ("scaler",StandardScaler()),
          ("model2",model2),
     ])
     pip_classifier.fit(X2_train,Y2_train)
     prediction_classifier=pip_classifier.predict(X2_test)
     print(prediction_classifier)

     # evaluation
     accuracy=accuracy_score(Y2_test,prediction_classifier)
     print(accuracy)
     precision=precision_score(Y2_test,prediction_classifier)
     print(precision)
     recall=recall_score(Y2_test,prediction_classifier)
     print(recall)
     f1score=f1_score(Y2_test,prediction_classifier)
     print(f1score)
     confusionmatrix=confusion_matrix(Y2_test,prediction_classifier)
     print(confusionmatrix)
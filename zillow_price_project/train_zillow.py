import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv("./train_2017.csv", parse_dates=["transactiondate"])
train_df = train_data
train_df['transaction_month'] = train_df['transactiondate'].dt.month
print("loading data.............")
prop_df = pd.read_csv("properties_2017.csv")
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
mean_values = train_df.mean(axis=0)
train_df.fillna(mean_values, inplace=True)



train_df['bedroomcnt'].ix[train_df['bedroomcnt']>8] = 8
train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
feat_names = train_df.columns.values


print("training.............")
y_train = train_y[int(len(train_df)*0.1): len(train_df)]
X_train = train_df[int(len(train_df)*0.1): len(train_df)]
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=20, random_state=0, loss='ls')
model.fit(X_train, y_train)



print("evaluating............")
(test_df, test_y) = (train_df[0:int(len(train_df)*0.1)], train_y[0:int(len(train_df)*0.1)])
y_test = test_y
X_test = test_df
print(mean_squared_error(y_test, model.predict(X_test)))







